# Prompt: run the whole Practice 3 project (football analytics with YOLO)

You are running a university project end to end on this machine. Do every step yourself. STOP and hand over to the human only when (a) a step needs a human (mouse clicks, manual annotation, a choice that is theirs, credentials), or (b) something fails and you cannot fix its CAUSE. Talk to the human in Brazilian Portuguese; keep code, commits and file contents as the project already has them.

## 0. Ground rules
1. Resume, never redo. Before each step, check whether its outputs already exist and are valid (files, `git log`). Heavy steps (tracking, camera) must not be rerun if their outputs are good.
2. Honesty: never invent or "estimate" a metric. Report every number exactly as a program printed it, including bad ones. Never tune a parameter just to make a number look better; change a parameter only with a stated technical reason.
3. Fix causes, not symptoms: no try/except that swallows errors, no hardcoded fixes.
4. Long commands (more than a few minutes): run them in the background, wait for completion, and kill any process you started that is left over (check the process list).
5. Never delete files with rm; move them to the recycle bin/trash, or rename them.
6. Git: commit at each milestone below. Commit messages must NOT contain Co-Authored-By or any other attribution line. Push only after the human agrees (the repo is public).
7. Stop format. When you stop, write: what is done, the exact state (files, numbers), what the human must do (numbered, with ready-to-paste commands), and what you will do after.

## 1. Project and repository
- Course: CESAR School, Computer Vision, Practice 3. Repo: https://github.com/LucasRafaelG/An-lise-de-Futebol-com-YOLO (branch main). Code in vc-20262-p3/. Read vc-20262-p3/ROTEIRO.md (assignment, deliverables, grading) and vc-20262-p3/README.md (commands, file formats) before starting. They are in Portuguese.
- Pipeline (one output folder per run, e.g. vc-20262-p3/saidas/jogo2):
  baixar_video.py -> detectar_rastrear.py -> remover_trechos.py (optional) -> bola_kalman.py -> homografia.py calibrar -> homografia.py camera -> homografia.py projetar -> metricas.py -> validar_passes.py -> xpass.py
- Run every command from the repository ROOT: meta.json stores the video path relative to it.

## 2. Environment
1. Clone or `git pull`. Run `git lfs install` and `git lfs pull`; check that the .mp4 files are real videos (tens of MB), not LFS pointer text files.
2. Create a venv and run `pip install -r vc-20262-p3/requirements.txt`.
3. ffmpeg must be on PATH (needed to download clips). On Windows: `winget install Gyan.FFmpeg`, then reload PATH in the current shell (a shell opened before the install will not see it).
4. Check the hardware: `python -c "import torch; print(torch.cuda.is_available())"`. Report GPU or CPU to the human.

## 3. Video (skip if a clip and its tracking already exist)
1. Need: 3 to 5 minutes from the MAIN wide side camera, open play, with no replays, close-ups or other angles. Broadcast footage is full of them, and every cut breaks the camera-motion step.
2. If the human has not given a URL and interval: STOP and ask for them.
3. Download: `python vc-20262-p3/baixar_video.py "URL" --inicio MIN --fim MIN --saida vc-20262-p3/videos`
4. Before tracking, build a contact sheet (e.g. `ffmpeg -i clip.mp4 -vf "fps=1/10,scale=480:-1,tile=4x6" -frames:v 1 sheet.jpg`) and look at it. List every non-main-camera stretch (replays, close-ups, behind-the-goal angles) with start and end times. If more than about 20% of the clip is like that, tell the human and suggest another interval. If they keep the clip, the stretches are removed in step 5.
5. Copyright: the repo is public and the clip is a broadcast. Ask the human before committing any video. If they agree, .mp4 files go through Git LFS (`git lfs track "*.mp4"`); GitHub rejects files over 100 MB without LFS.

## 4. Detection and tracking (heaviest step)
1. Measured on this project: yolov8m, imgsz 1280, CPU ran at 0.68 frames/s, so a 4-minute clip (7200 frames) took about 3 h. With a GPU use `--modelo yolov8m.pt --imgsz 1280`. On CPU use `--modelo yolov8s.pt --imgsz 640 --sem-video`, or a shorter window. Test 30 s first (`--inicio 0 --fim 0.5`).
2. `python vc-20262-p3/detectar_rastrear.py --video "vc-20262-p3/videos/<clip>.mp4" --saida vc-20262-p3/saidas/jogoN --modelo ...`
3. If Ultralytics auto-installs a missing package (e.g. `lap`) and asks for a restart, rerun the same command.
4. Outputs: tracks.csv, meta.json, frame0.jpg, rastreado.mp4. Commit tracks.csv and meta.json; rastreado.mp4 only via LFS and with the human's OK.

## 5. Remove bad stretches and pick a calibration frame
1. Remove the non-main-camera stretches from step 3.4 without redoing YOLO. remover_trechos.py keeps a backup in tracks_completo.csv. Times are m:ss from the clip start:
   `python vc-20262-p3/remover_trechos.py --saida vc-20262-p3/saidas/jogoN --excluir 0:40-1:00 1:30-1:48`
2. frame0.jpg (first frame of the run) MUST show at least 4 non-collinear field landmarks from campo.MARCOS (corners, halfway line with the touchlines, centre circle crossings and its left/right extremes circulo_esq/circulo_dir, penalty-box corners, penalty spots). Look at it. If it does not, find the first nearby frame that does, and create a new run starting there, reusing the tracking:
   `python vc-20262-p3/remover_trechos.py --saida vc-20262-p3/saidas/jogoN --destino vc-20262-p3/saidas/jogoM --inicio 0:05 --excluir <same stretches, original clip time>`
   This renumbers frames, shifts the excluded stretches, and writes a new meta.json and frame0.jpg.
3. Run `python vc-20262-p3/bola_kalman.py --saida <run>` and report its four printed numbers.
4. Commit.

## 6. Calibration (HUMAN STEP: STOP here)
1. Choose 5 or more well-spread, non-collinear landmarks visible in frame0.jpg. Give approximate pixel positions for each, and the exact command:
   `python vc-20262-p3/homografia.py calibrar --saida <run> --pontos name1,name2,...`
   A window opens: click the landmarks in the listed order; `u` undoes the last click, `q` finishes. Explain "sup/inf" (top/bottom touchline in the image) and "esq/dir" (left/right as seen from the main camera).
2. After the human confirms: read calibracao.json (per-point error in metres) and look at calibracao_grade.jpg (the 5 m grid must follow the pitch lines). The code fits by least squares when there are fewer than 8 points (RANSAC with 5 points would just drop one good click). If the grid is off, ask the human to redo the clicks; `calibrar --sem-janela` recomputes from saved clicks without the window.
3. Commit.

## 7. Camera motion (known problem: verify before trusting)
1. History: the original `homografia.py camera` chained per-frame homographies and produced garbage. On this project, after about 30 s the image centre of every frame mapped to the same point (degenerate matrices), and on match failure it fell back to the identity. A fix was being developed. Check the current homografia.py: the fix is present if there is a `_plausivel` function, invalid frames are stored as NaN, and projetar skips NaN frames. If it is missing, implement this spec:
   a. `_plausivel(M, w, h)`: M finite; the image corners mapped by M form a convex quadrilateral with the SAME orientation (signed area > 0, which rejects mirrored matrices); area ratio between 0.3 and 3.5.
   b. Accept a match when inliers >= floor AND `_plausivel`. Floors: 40 against the keyframe, 20 against recent frames, 40 against the keyframe bank. Do NOT add an inlier-ratio rule: a 0.5 ratio cut rejected clean 234-inlier matches and stranded the chain.
   c. References, tried in order: the current keyframe; then the last 6 valid frames, newest first (a deque), and a recent frame that matches becomes the new keyframe; then, every 3rd consecutive failure, a bank of up to 60 old keyframes spaced at least 15 frames apart (frame 0 always kept), to reconnect after a cut.
   d. A frame that matches nothing gets G = NaN (invalid); never the identity. Normalize every accepted G by G[2,2].
   e. projetar skips frames whose G is not finite (positions in metres stay NaN).
   f. Report the number of keyframes, the invalid % and the invalid stretches (>= 0.5 s) in seconds, and save them in camera_cortes.json next to the detected cuts.
2. Run: `python vc-20262-p3/homografia.py camera --saida <run>` (about 4 to 12 min).
3. Sanity check, always. Divide each valid G by G[2,2] and map the image centre (640, 360) to frame 0. For each 10 s window, print the valid frame count, the x/y range and the largest jump between consecutive valid frames. Healthy means a smooth pan, not constant, not absurd. Invalid frames should sit around detected cuts and removed stretches.
4. If it is unhealthy: STOP, report the numbers, and propose one of: (a) a targeted code fix with its reason; (b) split the run into one segment per main-camera stretch, each calibrated separately (more human clicks); (c) `metricas.py --pixels` with no homography, an approximation where xpass.py must not be used and the limitation must be discussed in the report.
5. If it is healthy: commit homografia.py, camera.npy, camera_cortes.json.

## 8. Projection and metrics
1. `python vc-20262-p3/homografia.py projetar --saida <run>`. Report the % of valid players and ball positions.
2. `python vc-20262-p3/metricas.py --saida <run>`. It skips speed jumps over gaps longer than 3 frames and passes longer than --dur-max, so removed stretches do not create fake passes. Report every printed number. Spot-check plausibility (top speeds under ~36 km/h; distances in a 4-minute clip on the order of a few hundred metres per player). If something is implausible, investigate the cause before going on.
3. Commit.

## 9. Pass validation (HUMAN STEP: STOP here)
1. Read the passes_anotados.csv format in README.md ("Formato dos arquivos"). Prepare an empty template with the right columns and tell the human: watch rastreado.mp4 and log every real pass. Warn about time offsets: if the run starts later than the clip (e.g. --inicio 0:05), rastreado.mp4 time = run time + offset.
2. After the human delivers it: `python vc-20262-p3/validar_passes.py --saida <run> --anotado passes_anotados.csv`. Report precision/recall exactly as printed.

## 10. eXP
`python vc-20262-p3/xpass.py --saida <run> --what-if`. Report the outputs, and do not use it if the run is in pixel mode. Commit.

## 11. Deliverable
Follow "Entrega" and "Critérios de avaliação" in ROTEIRO.md. Write the report in Portuguese, in the human's voice: plain, no hype words. Cover the clip URL and interval, the pipeline, every metric as printed, and the limitations: replays removed and how much, calibration error, % of invalid camera frames, ball coverage, and pass-validation precision/recall. Use only numbers that exist in the output files. STOP and give the human the report for review before any final commit/push.
