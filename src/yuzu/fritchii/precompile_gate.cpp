// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/fritchii/precompile_gate.h"

#include <algorithm>

#include <QDialog>
#include <QEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QLocale>
#include <QMainWindow>
#include <QProgressBar>
#include <QPushButton>
#include <QStatusBar>
#include <QTimer>
#include <QVBoxLayout>

#include "core/core.h"
#include "qt_common/qt_common.h"
#include "qt_common/render/emu_thread.h"
#include "video_core/gpu.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/shader_notify.h"
#include "yuzu/loading_screen.h"

namespace FritchiiUi {
namespace {

// A check that is over this quickly (prepared already, nothing to prepare) never shows the
// screen: the game only waits that long.
constexpr qint64 QuietCheckMs = 800;
// The rate behind "time left" comes from the last half minute.
constexpr qint64 RateWindowMs = 30'000;

VideoCore::ShaderNotify* Notify() {
    if (!QtCommon::system || !QtCommon::system->IsPoweredOn()) {
        return nullptr;
    }
    return &QtCommon::system->GPU().ShaderNotify();
}

} // namespace

PrecompileGate::PrecompileGate(QMainWindow* window_, QStatusBar* status_bar_,
                               PrecompileHooks hooks_)
    : QObject(window_), window{window_}, status_bar{status_bar_}, hooks{std::move(hooks_)} {
    timer = new QTimer(this);
    timer->setInterval(250);
    connect(timer, &QTimer::timeout, this, &PrecompileGate::Poll);

    status_note = new QLabel;
    status_note->setObjectName(QStringLiteral("FritchiiStatusNote"));
    status_note->hide();
    status_bar->addPermanentWidget(status_note);

    window->installEventFilter(this);
    clock.start();
}

PrecompileGate::~PrecompileGate() = default;

void PrecompileGate::OnEmulationStarting() {
    gated_this_boot = false;
    phase = Phase::Off;
    if (QtCommon::emu_thread) {
        connect(
            QtCommon::emu_thread.get(), &EmuThread::LoadProgress, this,
            [this](VideoCore::LoadCallbackStage stage, std::size_t, std::size_t) {
                if (stage == VideoCore::LoadCallbackStage::Complete) {
                    OnLoadComplete();
                }
            },
            Qt::QueuedConnection);
    }
}

void PrecompileGate::OnEmulationStopping() {
    if (auto* notify = Notify()) {
        notify->SetPrecompileFullSpeed(false);
    }
    timer->stop();
    phase = Phase::Off;
    status_note->hide();
    if (screen) {
        screen->hide();
        screen->deleteLater();
        screen = nullptr;
    }
}

void PrecompileGate::OnLoadComplete() {
    // Once per boot: a later shader cache reload says Complete too.
    if (gated_this_boot) {
        return;
    }
    gated_this_boot = true;
    auto* notify = Notify();
    if (!notify || !notify->PrecompilePending()) {
        return;
    }
    // Hold the game while the preparation decides; a quick "already prepared" lets it go at once.
    hooks.pause();
    phase = Phase::Checking;
    samples.clear();
    since_check.start();
    timer->start();
}

void PrecompileGate::Poll() {
    auto* notify = Notify();
    if (!notify) {
        OnEmulationStopping();
        return;
    }
    const bool pending = notify->PrecompilePending();
    const auto progress = notify->GetPrecompileProgress();
    const bool counting = progress.total > 0 && progress.done < progress.total;
    if (counting) {
        samples.emplace_back(clock.elapsed(), progress.done);
        while (samples.size() > 2 && samples.back().first - samples.front().first > RateWindowMs) {
            samples.pop_front();
        }
    }

    switch (phase) {
    case Phase::Checking:
        if (!pending) {
            phase = Phase::Off;
            timer->stop();
            hooks.resume();
        } else if (counting || since_check.elapsed() > QuietCheckMs) {
            ShowScreen();
            UpdateScreen(progress.done, progress.total);
        }
        break;
    case Phase::Screen:
        if (!pending) {
            CloseScreen();
            phase = Phase::Off;
            timer->stop();
            hooks.resume();
        } else {
            UpdateScreen(progress.done, progress.total);
        }
        break;
    case Phase::Background:
        if (!pending) {
            status_note->hide();
            phase = Phase::Off;
            timer->stop();
        } else if (counting) {
            const int percent = static_cast<int>(progress.done * 100 / progress.total);
            status_note->setText(tr("Preparing shaders %1%").arg(percent));
            status_note->show();
        }
        break;
    case Phase::Off:
        timer->stop();
        break;
    }
}

void PrecompileGate::ShowScreen() {
    phase = Phase::Screen;
    if (auto* notify = Notify()) {
        notify->SetPrecompileFullSpeed(true);
    }
    if (screen) {
        screen->show();
        return;
    }
    screen = new QDialog(window, Qt::FramelessWindowHint | Qt::Dialog);
    screen->setObjectName(QStringLiteral("FritchiiPrepare"));
    screen->setAttribute(Qt::WA_StyledBackground);
    screen->setModal(true);
    // Esc or Alt+F4 on the screen: play now (the preparation goes on in the background).
    connect(screen, &QDialog::rejected, this, &PrecompileGate::PlayNow);

    auto* outer = new QVBoxLayout(screen);
    outer->addStretch(1);
    auto* column_row = new QHBoxLayout;
    column_row->addStretch(1);
    auto* column = new QVBoxLayout;
    column->setSpacing(0);

    auto* title = new QLabel(tr("Preparing shaders"));
    title->setObjectName(QStringLiteral("FritchiiHeroTitle"));
    column->addWidget(title);
    column->addSpacing(14);

    sentence = new QLabel(tr("This happens once, so %1 runs smoothly from the start.")
                              .arg(hooks.game_name()));
    sentence->setObjectName(QStringLiteral("FritchiiLead"));
    sentence->setWordWrap(true);
    column->addWidget(sentence);
    column->addSpacing(32);

    bar = new QProgressBar;
    bar->setObjectName(QStringLiteral("FritchiiPrepareBar"));
    bar->setTextVisible(false);
    bar->setFixedHeight(12);
    column->addWidget(bar);
    column->addSpacing(12);

    auto* numbers = new QHBoxLayout;
    count = new QLabel;
    count->setObjectName(QStringLiteral("FritchiiLead"));
    eta = new QLabel;
    eta->setObjectName(QStringLiteral("FritchiiLead"));
    numbers->addWidget(count);
    numbers->addStretch(1);
    numbers->addWidget(eta);
    column->addLayout(numbers);
    column->addSpacing(40);

    auto* play = new QPushButton(tr("Play now"));
    play->setMinimumSize(150, 44);
    play->setCursor(Qt::PointingHandCursor);
    connect(play, &QPushButton::clicked, this, &PrecompileGate::PlayNow);
    column->addWidget(play, 0, Qt::AlignLeft);

    auto* column_box = new QWidget;
    column_box->setLayout(column);
    column_box->setFixedWidth(560);
    column_row->addWidget(column_box);
    column_row->addStretch(1);
    outer->addLayout(column_row);
    outer->addStretch(1);

    PlaceScreen();
    screen->show();
    screen->raise();
    screen->activateWindow();
}

void PrecompileGate::UpdateScreen(size_t done, size_t total) {
    if (!screen) {
        return;
    }
    const QLocale locale;
    if (total == 0) {
        // Looking for the file or assembling it from the recipe: no count yet.
        bar->setRange(0, 0);
        count->setText(tr("Getting ready"));
        eta->clear();
        return;
    }
    // QProgressBar takes an int: scale large counts.
    bar->setRange(0, 1000);
    bar->setValue(static_cast<int>(done * 1000 / total));
    count->setText(tr("%1 / %2").arg(locale.toString(static_cast<qulonglong>(done)),
                                      locale.toString(static_cast<qulonglong>(total))));
    eta->setText(TimeLeft(done, total));
}

QString PrecompileGate::TimeLeft(size_t done, size_t total) {
    if (samples.size() < 2) {
        return {};
    }
    const auto& [first_ms, first_done] = samples.front();
    const auto& [last_ms, last_done] = samples.back();
    const qint64 span = last_ms - first_ms;
    if (span < 3000 || last_done <= first_done) {
        return {};
    }
    const double per_ms = static_cast<double>(last_done - first_done) / static_cast<double>(span);
    const double seconds = static_cast<double>(total - done) / per_ms / 1000.0;
    if (seconds < 60) {
        return tr("Less than a minute left");
    }
    const int minutes = static_cast<int>(seconds / 60 + 0.5);
    if (minutes < 60) {
        return tr("About %n min left", "", minutes);
    }
    return tr("About %1 h %2 min left").arg(minutes / 60).arg(minutes % 60);
}

void PrecompileGate::PlayNow() {
    if (phase != Phase::Screen) {
        return;
    }
    if (auto* notify = Notify()) {
        notify->SetPrecompileFullSpeed(false);
    }
    CloseScreen();
    phase = Phase::Background;
    hooks.resume();
}

void PrecompileGate::CloseScreen() {
    if (auto* notify = Notify()) {
        notify->SetPrecompileFullSpeed(false);
    }
    if (screen) {
        QDialog* closing = screen;
        screen = nullptr;
        closing->hide();
        closing->deleteLater();
    }
}

void PrecompileGate::PlaceScreen() {
    if (!screen) {
        return;
    }
    QWidget* area = window->centralWidget();
    const QRect rect(area->mapToGlobal(QPoint(0, 0)), area->size());
    screen->setGeometry(rect);
}

bool PrecompileGate::eventFilter(QObject* watched, QEvent* event) {
    if (watched == window && screen &&
        (event->type() == QEvent::Move || event->type() == QEvent::Resize)) {
        PlaceScreen();
    }
    return QObject::eventFilter(watched, event);
}

} // namespace FritchiiUi
