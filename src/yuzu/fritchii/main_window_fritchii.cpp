// SPDX-License-Identifier: GPL-3.0-or-later

// MainWindow's Fritchii setup, kept out of main_window.cpp so Lichii merges stay clean. It runs
// once at the end of the constructor and does nothing in a Lichii build.

#include <QApplication>
#include <QStatusBar>
#include <QTimer>

#include "core/core.h"
#include "qt_common/qt_common.h"
#include "ui_main.h"
#include "yuzu/fritchii/fritchii_ui.h"
#include "yuzu/fritchii/home.h"
#include "yuzu/fritchii/precompile_gate.h"
#include "yuzu/fritchii/whats_new.h"
#include "yuzu/game/game_list.h"
#include "yuzu/main_window.h"
#include "yuzu/multiplayer/state.h"
#include "yuzu/util/clickable_label.h"

void MainWindow::SetUpFritchiiUi() {
    if (!FritchiiUi::Enabled()) {
        return;
    }

    FritchiiUi::RestructureMenus(*ui, *this,
                                 {.export_log = [this] { OnExportLog(); },
                                  .whats_new = [this] { FritchiiUi::ShowWhatsNew(this); }});

    // "What's new" once per version after an update, when nothing else is on screen: no other
    // dialog (first-launch question, update offer, restart note) and no game running.
    const bool used_before = Settings::values.oot_fps_asked.GetValue();
    if (FritchiiUi::ShouldShowWhatsNew(used_before)) {
        auto* timer = new QTimer(this);
        timer->setInterval(1500);
        connect(timer, &QTimer::timeout, this, [this, timer] {
            const bool first_question_pending =
                !launched_with_game && !Settings::values.oot_fps_asked.GetValue();
            if (QApplication::activeModalWidget() != nullptr || first_question_pending ||
                emulation_running || boot_in_progress) {
                return;
            }
            timer->stop();
            timer->deleteLater();
            FritchiiUi::MarkWhatsNewSeen();
            OnSaveConfig();
            FritchiiUi::ShowWhatsNew(this);
        });
        timer->start();
    } else if (!used_before) {
        OnSaveConfig(); // a first install recorded its version
    }

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

    // "Preparing shaders" before the game starts when its precompiled pipelines need preparing.
    auto* gate = new FritchiiUi::PrecompileGate(
        this, statusBar(),
        {.pause = [this] { OnPauseGame(); },
         .resume = [this] { OnStartGame(); },
         .game_name = [] {
             constexpr u64 OcarinaOfTime = 0x0400486024DEC000ULL;
             return QtCommon::system &&
                            QtCommon::system->GetApplicationProcessProgramID() == OcarinaOfTime
                        ? QStringLiteral("OoT")
                        : tr("the game");
         }});
    connect(this, &MainWindow::EmulationStarting, gate,
            &FritchiiUi::PrecompileGate::OnEmulationStarting);
    connect(this, &MainWindow::EmulationStopping, gate,
            &FritchiiUi::PrecompileGate::OnEmulationStopping);
}
