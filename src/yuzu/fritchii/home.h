// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <string>
#include <QWidget>
#include "common/common_types.h"

class QBoxLayout;
class QHBoxLayout;
class QLabel;
class QMainWindow;
class QMenu;
class QPushButton;
class QStandardItemModel;
class QTimer;
class QToolButton;

namespace FritchiiUi {

/// What the home screen asks the main window to do.
struct HomeHooks {
    std::function<void(const QString& path, u64 program_id)> play;
    std::function<void()> add_folder;
    std::function<void()> open_settings;
    std::function<void(u64 program_id, const std::string& path)> game_settings;
    std::function<bool()> game_list_empty;
    std::function<bool()> single_window;
    std::function<bool()> emulation_running;
    QStandardItemModel* model = nullptr;
    QMenu* help_menu = nullptr;
    QList<QMenu*> classic_menus;  ///< File, Game, View, Tools, Help behind the rail's menu button
};

/// The Fritchii main screen: a left icon rail (Library, Settings, Help, and the classic menus
/// behind a button) and a home page with one big card for Ocarina of Time (or the first game)
/// and the other games in a small row. The classic game list stays one click away
/// ("All games"); the main window keeps managing it as before.
class Home : public QObject {
    Q_OBJECT
public:
    Home(QMainWindow* window, QBoxLayout* central, QWidget* game_list, QWidget* placeholder,
         HomeHooks hooks);

    /// A game is starting in the main window: the render view takes the space.
    void OnEmulationStarting();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void ShowHome();
    void ShowList();
    void Restore();
    void Refresh();

    QWidget* MakeRail();
    QWidget* MakePage();
    QToolButton* RailButton(const QString& icon, const QString& tip);

    QMainWindow* window;
    QWidget* game_list;
    QWidget* placeholder;
    HomeHooks hooks;

    QWidget* rail = nullptr;
    QWidget* page = nullptr;
    QToolButton* library_button = nullptr;
    QLabel* cover = nullptr;
    QLabel* title = nullptr;
    QPushButton* play = nullptr;
    QHBoxLayout* row = nullptr;
    QWidget* row_box = nullptr;
    QTimer* refresh_timer = nullptr;

    bool list_mode = false;
    QString hero_path;
    u64 hero_id = 0;
};

} // namespace FritchiiUi
