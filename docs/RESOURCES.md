# Runtime resources

Use this project's `.venv`; never borrow an interpreter from another project.
Set `DAFC_PROJECT_ROOT` to the workspace when invoking the installed package
outside the repository. Compatibility launchers set it to their repository.
Generated runs belong in `data/` or `outputs/`, not in the package installation.

Offline routing needs only the pinned NumPy, OpenCV and Pillow dependencies.
Browser experiments additionally need MiniWoB++, a compatible Chromium/WebDriver,
and optionally Playwright. Install the browser extra in a separate environment
when running that benchmark. Android execution needs AndroidWorld commit
`3e50888527ef9f29b9157ecd537e408008bb1c85`, its declared Android environment
dependencies, an emulator and Android SDK (`ANDROID_HOME`). The accessibility
forwarder APK belongs at `resources/external/2024.05.13-accessibility_forwarder.apk`
and must hash to `97a56a544e44d79f9b3181fc7dbdd72cffa908efd3d53c82afad1773061a350a`.
Do not install/recreate the retired server's entire environment.

Actual actors: Qwen2.5-VL-7B-Instruct and UI-TARS-2B-SFT. Supply model checkpoints
and explicit service URLs to the retained model runners. Their upstream model
licenses and Android/MiniWoB licenses apply. GPU inference and full live browser/
emulator experiments were not rerun in this migration; exact frozen runtime
records remain with the private evidence. Core tests do not stand in for them.

The sealed source registry and its image receipts are external experiment
inputs. Supply them explicitly under `data/`; their content hashes are checked.
The retired Android V2 analysis adapter requires its frozen analyzer checkout,
not the current package. It fails if those sources are absent. Its pure integral
coordinate repair is available independently as `dafc.coordinates`.

No hosting URL for large raw evidence has been published. Request it from the
owner; locally it remains in `_research_archive`, with paths in the private
delivery report. This is an explicit remaining release resource task.
