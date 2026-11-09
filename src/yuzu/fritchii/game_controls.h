// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>

#include <QElapsedTimer>
#include <QList>
#include <QWidget>

class QAction;
class QMenu;
class QTimer;
class QToolButton;

namespace FritchiiUi {

struct GameControlHooks {
    std::function<QWidget*()> render;   ///< the game's render widget
    std::function<bool()> running;      ///< the game runs (not paused)
    QAction* pause = nullptr;           ///< Pause / Continue
    QAction* restart = nullptr;
    QAction* stop = nullptr;
    QAction* fullscreen = nullptr;      ///< checkable
    QList<QMenu*> menus;                ///< the classic menus, behind the last button
};

/// The in-game control strip: a small Graphite bar at the top centre of the game that shows when
/// the mouse comes near the top edge, with Pause/Continue, Restart, Stop, Fullscreen and the
/// classic menus. It hides again a moment after the mouse leaves, never takes the keyboard focus
/// and stays away while a dialog is open.
class GameControls : public QWidget {
    Q_OBJECT
public:
    GameControls(QWidget* owner, GameControlHooks hooks);

    void Start();
    void Stop();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void Tick();
    void Place(const QRect& area);
    QToolButton* Button(const QString& icon, const QString& tip, QAction* action);

    GameControlHooks hooks;
    QTimer* timer = nullptr;
    QToolButton* pause_button = nullptr;
    QToolButton* fullscreen_button = nullptr;
    QMenu* classic = nullptr;
    QElapsedTimer since_hot;
    bool shown_running = true;
    bool shown_fullscreen = false;
};

} // namespace FritchiiUi
