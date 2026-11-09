// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/fritchii/game_controls.h"

#include <QAction>
#include <QApplication>
#include <QCursor>
#include <QHBoxLayout>
#include <QMenu>
#include <QPainter>
#include <QTimer>
#include <QToolButton>

#include "yuzu/fritchii/fritchii_ui.h"

namespace FritchiiUi {
namespace {

constexpr Qt::WindowFlags StripFlags = Qt::Tool | Qt::FramelessWindowHint |
                                       Qt::WindowDoesNotAcceptFocus |
                                       Qt::NoDropShadowWindowHint;
// The strip shows while the mouse is this close to the top of the game, and stays this long
// after it leaves (or leaves the strip).
constexpr int HotZone = 90;
constexpr qint64 LingerMs = 1200;
constexpr unsigned IconColor = 0xececf1;

} // namespace

GameControls::GameControls(QWidget* owner, GameControlHooks hooks_)
    : QWidget(owner, StripFlags), hooks{std::move(hooks_)} {
    setObjectName(QStringLiteral("FritchiiStrip"));
    setAttribute(Qt::WA_TranslucentBackground);
    setAttribute(Qt::WA_ShowWithoutActivating);

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(10, 8, 10, 8);
    layout->setSpacing(4);

    pause_button = Button(QStringLiteral("pause"), tr("Pause"), hooks.pause);
    layout->addWidget(pause_button);
    layout->addWidget(Button(QStringLiteral("restart"), tr("Restart"), hooks.restart));
    layout->addWidget(Button(QStringLiteral("stop"), tr("Stop"), hooks.stop));
    fullscreen_button = Button(QStringLiteral("fullscreen"), tr("Fullscreen"), hooks.fullscreen);
    layout->addWidget(fullscreen_button);

    classic = new QMenu(this);
    for (QMenu* menu : hooks.menus) {
        classic->addMenu(menu);
    }
    auto* menu_button = Button(QStringLiteral("menu"), tr("Menu"), nullptr);
    connect(menu_button, &QToolButton::clicked, this, [this, menu_button] {
        classic->exec(menu_button->mapToGlobal(QPoint(0, menu_button->height() + 6)));
    });
    layout->addWidget(menu_button);
    adjustSize();

    timer = new QTimer(this);
    timer->setInterval(100);
    connect(timer, &QTimer::timeout, this, &GameControls::Tick);
}

QToolButton* GameControls::Button(const QString& icon, const QString& tip, QAction* action) {
    auto* button = new QToolButton(this);
    button->setObjectName(QStringLiteral("FritchiiStripButton"));
    button->setIcon(Glyph(icon, IconColor, 22));
    button->setIconSize(QSize(22, 22));
    button->setFixedSize(44, 44);
    button->setToolTip(tip);
    button->setFocusPolicy(Qt::NoFocus);
    button->setCursor(Qt::PointingHandCursor);
    if (action) {
        connect(button, &QToolButton::clicked, action, &QAction::trigger);
    }
    return button;
}

void GameControls::Start() {
    since_hot.invalidate();
    timer->start();
}

void GameControls::Stop() {
    timer->stop();
    hide();
}

void GameControls::Tick() {
    QWidget* render = hooks.render ? hooks.render() : nullptr;
    const bool usable = render && render->isVisible() &&
                        QApplication::applicationState() == Qt::ApplicationActive &&
                        QApplication::activeModalWidget() == nullptr &&
                        QApplication::activePopupWidget() == nullptr;
    if (!usable) {
        if (!classic->isVisible()) {
            hide();
        }
        return;
    }
    // A separate game window (single window mode off) owns the strip, so it stays above it.
    if (parentWidget() != render->window()) {
        setParent(render->window(), StripFlags);
    }

    const QRect area(render->mapToGlobal(QPoint(0, 0)), render->size());
    const QPoint cursor = QCursor::pos();
    const QRect hot(area.left(), area.top(), area.width(), HotZone);
    if (hot.contains(cursor) || (isVisible() && frameGeometry().contains(cursor))) {
        since_hot.restart();
    }
    const bool wanted = since_hot.isValid() && since_hot.elapsed() < LingerMs;
    if (!wanted) {
        hide();
        return;
    }

    const bool running = hooks.running ? hooks.running() : true;
    if (running != shown_running || !isVisible()) {
        shown_running = running;
        pause_button->setIcon(Glyph(running ? QStringLiteral("pause") : QStringLiteral("play"),
                                    IconColor, 22));
        pause_button->setToolTip(running ? tr("Pause") : tr("Continue"));
    }
    const bool fullscreen = hooks.fullscreen && hooks.fullscreen->isChecked();
    if (fullscreen != shown_fullscreen || !isVisible()) {
        shown_fullscreen = fullscreen;
        fullscreen_button->setIcon(Glyph(
            fullscreen ? QStringLiteral("fullscreen-exit") : QStringLiteral("fullscreen"),
            IconColor, 22));
        fullscreen_button->setToolTip(fullscreen ? tr("Leave fullscreen") : tr("Fullscreen"));
    }
    Place(area);
    if (!isVisible()) {
        show();
    }
    raise();
}

void GameControls::Place(const QRect& area) {
    const QSize size = sizeHint();
    move(area.left() + (area.width() - size.width()) / 2, area.top() + 14);
}

void GameControls::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(QColor(255, 255, 255, 22));
    painter.setBrush(QColor(0x16, 0x16, 0x1a, 235));
    painter.drawRoundedRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5), 14, 14);
}

} // namespace FritchiiUi
