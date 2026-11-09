// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/fritchii/perf_overlay.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <QFont>
#include <QFontMetrics>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>

#include "common/settings.h"
#include "core/perf_stats.h"
#include "yuzu/main_window.h"

namespace FritchiiUi {
namespace {

constexpr size_t MaxSamples = 90;
// A sample this low is a freeze, a loading screen or the boot: a gap in the line, not a dip.
constexpr double GapFps = 10.0;
// "low" is the lowest of the most recent samples (about half a minute).
constexpr size_t LowSamples = 30;
constexpr int Width = 272;
constexpr int Height = 158;
constexpr int Pad = 18;
constexpr int GraphHeight = 54;

const QColor Panel{0x16, 0x16, 0x1a, 232};
const QColor Edge{255, 255, 255, 22};
const QColor Text{0xec, 0xec, 0xf1};
const QColor Muted{0x9a, 0x9a, 0xa8};
const QColor Accent{0xa7, 0x8b, 0xfa};
const QColor Guide{255, 255, 255, 34};

QFont InterFont(int pixel_size, QFont::Weight weight) {
    QFont font(QStringLiteral("Inter"));
    font.setPixelSize(pixel_size);
    font.setWeight(weight);
    return font;
}

} // namespace

PerfOverlay::PerfOverlay(MainWindow* main_window_, std::function<void()> on_hidden_)
    : QWidget(main_window_, Qt::Tool | Qt::FramelessWindowHint | Qt::WindowDoesNotAcceptFocus |
                                Qt::NoDropShadowWindowHint),
      main_window{main_window_}, on_hidden{std::move(on_hidden_)} {
    setAttribute(Qt::WA_TranslucentBackground);
    setAttribute(Qt::WA_ShowWithoutActivating);
    setFixedSize(Width, Height);
    setCursor(Qt::OpenHandCursor);
    FollowWindow();
    connect(main_window, &MainWindow::positionChanged, this, &PerfOverlay::FollowWindow);
    connect(main_window, &MainWindow::statsUpdated, this,
            [this](const Core::PerfStatsResults& results, const VideoCore::ShaderNotify&) {
                OnStats(results);
            });
}

void PerfOverlay::FollowWindow() {
    move(main_window->pos() + offset);
}

void PerfOverlay::OnStats(const Core::PerfStatsResults& results) {
    if (!std::isnan(results.average_game_fps)) {
        fps = results.average_game_fps;
        fps_samples.push_back(fps > GapFps ? fps : std::numeric_limits<double>::quiet_NaN());
        while (fps_samples.size() > MaxSamples) {
            fps_samples.pop_front();
        }
    }
    if (!std::isnan(results.frametime)) {
        frametime_ms = results.frametime * 1000.0;
    }
    update();
}

void PerfOverlay::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(Edge);
    painter.setBrush(Panel);
    painter.drawRoundedRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5), 16, 16);

    // The frame rate, big, and "fps" beside it.
    const QFont big = InterFont(40, QFont::Medium);
    const QFont label = InterFont(15, QFont::Normal);
    const QString fps_text = QString::number(std::lround(fps));
    const int baseline = Pad + QFontMetrics(big).ascent() - 6;
    painter.setFont(big);
    painter.setPen(Text);
    painter.drawText(Pad, baseline, fps_text);
    const int fps_width = QFontMetrics(big).horizontalAdvance(fps_text);
    painter.setFont(label);
    painter.setPen(Muted);
    painter.drawText(Pad + fps_width + 8, baseline, tr("fps"));

    // Frame time and the lowest recent frame rate, right-aligned.
    double low = std::numeric_limits<double>::infinity();
    const size_t first = fps_samples.size() > LowSamples ? fps_samples.size() - LowSamples : 0;
    for (size_t i = first; i < fps_samples.size(); ++i) {
        if (!std::isnan(fps_samples[i])) {
            low = std::min(low, fps_samples[i]);
        }
    }
    const QFont value = InterFont(17, QFont::Medium);
    const QString frametime_text = tr("%1 ms").arg(frametime_ms, 0, 'f', 1);
    const QString low_text =
        std::isinf(low) ? QString{} : tr("low %1").arg(std::lround(low));
    const int right = Width - Pad;
    painter.setFont(value);
    painter.setPen(Text);
    painter.drawText(QRect(right - 140, Pad, 140, 24), Qt::AlignRight | Qt::AlignVCenter,
                     frametime_text);
    painter.setFont(label);
    painter.setPen(Muted);
    painter.drawText(QRect(right - 140, Pad + 26, 140, 20), Qt::AlignRight | Qt::AlignVCenter,
                     low_text);

    // The line: the last samples against the target frame rate.
    const QRectF graph(Pad, Height - Pad - GraphHeight, Width - 2 * Pad, GraphHeight);
    const double target = Settings::values.oot_30fps.GetValue() ? 30.0 : 60.0;
    double top = target * 1.25;
    for (const double sample : fps_samples) {
        if (!std::isnan(sample)) {
            top = std::max(top, sample * 1.1);
        }
    }
    const auto y_of = [&](double v) { return graph.bottom() - v / top * graph.height(); };
    painter.setPen(QPen(Guide, 1, Qt::DashLine));
    painter.drawLine(QPointF(graph.left(), y_of(target)), QPointF(graph.right(), y_of(target)));

    if (fps_samples.size() >= 2) {
        const double step = graph.width() / static_cast<double>(MaxSamples - 1);
        const double x0 = graph.right() - step * static_cast<double>(fps_samples.size() - 1);
        QPainterPath line;
        QPainterPath fill;
        bool open = false;
        double run_start = 0;
        double last_x = 0;
        const auto close_fill = [&] {
            if (open) {
                fill.lineTo(last_x, graph.bottom());
                fill.lineTo(run_start, graph.bottom());
                fill.closeSubpath();
            }
        };
        for (size_t i = 0; i < fps_samples.size(); ++i) {
            const double sample = fps_samples[i];
            const double x = x0 + step * static_cast<double>(i);
            if (std::isnan(sample)) {
                close_fill();
                open = false;
                continue;
            }
            const double y = y_of(std::min(sample, top));
            if (!open) {
                line.moveTo(x, y);
                fill.moveTo(x, y);
                run_start = x;
                open = true;
            } else {
                line.lineTo(x, y);
                fill.lineTo(x, y);
            }
            last_x = x;
        }
        close_fill();
        QColor area = Accent;
        area.setAlpha(46);
        painter.setPen(Qt::NoPen);
        painter.setBrush(area);
        painter.drawPath(fill);
        painter.setBrush(Qt::NoBrush);
        painter.setPen(QPen(Accent, 2.2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        painter.drawPath(line);
    }
}

void PerfOverlay::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::RightButton) {
        hide();
        if (on_hidden) {
            on_hidden();
        }
        return;
    }
    if (event->button() == Qt::LeftButton) {
        drag_start = event->pos();
    }
}

void PerfOverlay::mouseMoveEvent(QMouseEvent* event) {
    if (event->buttons() & Qt::LeftButton) {
        const QPoint position = event->globalPosition().toPoint() - drag_start;
        offset = position - main_window->pos();
        move(position);
    }
}

} // namespace FritchiiUi
