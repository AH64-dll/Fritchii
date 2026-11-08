// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <deque>
#include <functional>
#include <utility>

#include <QElapsedTimer>
#include <QObject>
#include <QString>

#include "common/common_types.h"

class QDialog;
class QLabel;
class QMainWindow;
class QProgressBar;
class QStatusBar;
class QTimer;

namespace FritchiiUi {

struct PrecompileHooks {
    std::function<void()> pause;           ///< hold the game (MainWindow::OnPauseGame)
    std::function<void()> resume;          ///< let it run (MainWindow::OnStartGame)
    std::function<QString()> game_name;    ///< "OoT", or "the game"
};

/// Fritchii's "Preparing shaders" screen (EDEN_NX2_PRECOMPILE). Right after a game has loaded, it
/// holds the game while the precompiled pipelines are looked for, assembled or prepared, and
/// shows a screen with the progress while the preparation runs at full speed. "Play now" lets the
/// game run and the preparation continue in the background; a status bar note shows it until it
/// is done. When nothing needs preparing the game starts as before.
class PrecompileGate : public QObject {
    Q_OBJECT
public:
    PrecompileGate(QMainWindow* window, QStatusBar* status_bar, PrecompileHooks hooks);
    ~PrecompileGate() override;

    void OnEmulationStarting();
    void OnEmulationStopping();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    enum class Phase { Off, Checking, Screen, Background };

    void OnLoadComplete();
    void Poll();
    void ShowScreen();
    void PlayNow();
    void CloseScreen();
    void PlaceScreen();
    void UpdateScreen(size_t done, size_t total);
    QString TimeLeft(size_t done, size_t total);

    QMainWindow* window;
    QStatusBar* status_bar;
    PrecompileHooks hooks;

    QTimer* timer = nullptr;
    QLabel* status_note = nullptr;
    QDialog* screen = nullptr;
    QProgressBar* bar = nullptr;
    QLabel* sentence = nullptr;
    QLabel* count = nullptr;
    QLabel* eta = nullptr;

    Phase phase = Phase::Off;
    bool gated_this_boot = false;
    QElapsedTimer since_check;
    QElapsedTimer clock;
    std::deque<std::pair<qint64, size_t>> samples;
};

} // namespace FritchiiUi
