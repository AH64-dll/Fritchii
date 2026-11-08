// SPDX-License-Identifier: GPL-3.0-or-later

// MainWindow's Fritchii setup, kept out of main_window.cpp so Lichii merges stay clean. It runs
// once at the end of the constructor and does nothing in a Lichii build.

#include <QStatusBar>

#include "core/core.h"
#include "qt_common/qt_common.h"
#include "ui_main.h"
#include "yuzu/fritchii/fritchii_ui.h"
#include "yuzu/fritchii/home.h"
#include "yuzu/game/game_list.h"
#include "yuzu/main_window.h"
#include "yuzu/multiplayer/state.h"
#include "yuzu/util/clickable_label.h"

void MainWindow::SetUpFritchiiUi() {
    if (!FritchiiUi::Enabled()) {
        return;
    }

    FritchiiUi::RestructureMenus(*ui, *this, {.export_log = [this] { OnExportLog(); }});

    // Status bar: the everyday controls stay (60/30 FPS, docked, volume); renderer, accuracy,
    // filter, anti-aliasing and effects move into one Graphics button. The old buttons stay
    // hidden: they still hold the state the rest of the window updates.
    for (QWidget* expert : {static_cast<QWidget*>(renderer_status_button),
                            static_cast<QWidget*>(gpu_accuracy_button),
                            static_cast<QWidget*>(filter_status_button),
                            static_cast<QWidget*>(aa_status_button),
                            static_cast<QWidget*>(post_shader_status_button),
                            static_cast<QWidget*>(refresh_button)}) {
        if (expert) {
            expert->hide();
        }
    }
    multiplayer_state->GetStatusText()->hide();
    FritchiiUi::GraphicsHooks graphics{
        .refresh =
            [this] {
                UpdateStatusButtons();
#ifdef HAS_RESHADE
                UpdatePostShaderText();
#endif
            },
        .renderer_locked = [this] { return !renderer_status_button->isEnabled(); },
        .open_settings = [this] { OnConfigure(); },
#ifdef HAS_RESHADE
        .open_effects = [this] { OnPostProcessingShaders(); },
#endif
    };
    statusBar()->insertPermanentWidget(0, FritchiiUi::MakeGraphicsButton(this, graphics));

    // Home screen and rail.
    FritchiiUi::HomeHooks home{
        .play = [this](const QString& path, u64 program_id) { OnGameListLoadFile(path, program_id); },
        .add_folder = [this] { OnGameListAddDirectory(); },
        .open_settings = [this] { OnConfigure(); },
        .game_settings =
            [this](u64 program_id, const std::string& path) {
                OpenPerGameConfiguration(program_id, path);
            },
        .game_list_empty = [this] { return game_list->IsEmpty(); },
        .single_window = [this] { return ui->action_Single_Window_Mode->isChecked(); },
        .emulation_running = [this] { return emulation_running; },
        .model = game_list->GetModel(),
        .help_menu = ui->menu_Help,
        .classic_menus = {ui->menu_File, ui->menu_Emulation, ui->menu_View, ui->menu_Tools,
                          ui->menu_Help},
    };
    fritchii_home = new FritchiiUi::Home(this, ui->horizontalLayout, game_list,
                                         game_list_placeholder, std::move(home));
    connect(this, &MainWindow::EmulationStarting, fritchii_home,
            &FritchiiUi::Home::OnEmulationStarting);
}
