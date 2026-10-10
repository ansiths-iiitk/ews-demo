# Putting the Early-Warning app online (one time, about 15 minutes)

You need a free GitHub account and a free Streamlit account (sign in to Streamlit with GitHub).

## 1. Put the files on GitHub
1. Go to github.com → **New repository**. Name it `ews-demo`, set it to **Public**, click **Create**.
2. On the new repository page click **uploading an existing file**.
3. Drag in every file from this folder: `app.py`, `core.py`, `train_model.py`, `model.joblib`,
   `training_data.csv`, `requirements.txt`, `.gitignore`. Click **Commit changes**.

## 2. Publish it
1. Go to share.streamlit.io → sign in with GitHub → **Create app** → **Deploy a public app from GitHub**.
2. Repository: `ews-demo`. Branch: `main`. Main file: `app.py`.
3. Choose a short app address, e.g. `iiitk-ews` (becomes `iiitk-ews.streamlit.app`).
4. Open **Advanced settings → Secrets** and paste (choose your own code):
   ```
   DASHBOARD_CODE = "your-own-code"
   ```
5. Click **Deploy**. The first start takes 2–3 minutes.

## 3. Your two links
- Participants: `https://iiitk-ews.streamlit.app`  (put this as the QR code on the slide)
- You:          `https://iiitk-ews.streamlit.app/?view=dashboard`  (enter your code)

## Before every class
- **Open both links 30 minutes early.** Free apps go to sleep when nobody uses them; opening the link wakes them.
- On the dashboard: set the release to **v1.0** and use **Reset** to clear rehearsal data.
- The log is lost if the app restarts, so press **Download tonight's data (CSV)** before Hands-on 2.

## During class: what to press
| Moment | On the dashboard |
|---|---|
| Opening and "break my model" | Release **v1.0** (no checking: nonsense gets confident answers; blanks crash) |
| After the break list | Switch to **v1.1** (checking added). Ask them to retry their breaks. |
| "Marks are now out of 20" | Stay on **v1.1**. Even roll numbers enter marks out of 20. Watch the drift box turn red and the flagged share rise. Point out: validation passed every one of these inputs. |
| New release | Switch to **v2.0**. Ask for normal inputs. Almost everyone is flagged, with no crash and no drift alarm. |
| Rollback | Switch back to **v1.1**. Ask what monitoring caught it (the flagged share line), and what did not (input checking, drift check). |

## If the link fails in class
Use the Colab notebook (coming next) and show the dashboard screenshots from your rehearsal.
