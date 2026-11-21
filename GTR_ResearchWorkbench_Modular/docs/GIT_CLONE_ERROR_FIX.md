# AutoDL Git index-pack failure — fix

If `git clone` prints `fatal: could not open ... tmp_pack...` or `index-pack failed`, it **does not necessarily mean you have no disk space**. The old installer used `git clone` first; this revision obtains the pinned GitHub source ZIP directly, and verifies official file hashes before installation.

From the project root:

```bash
bash 05_fix_official_download.sh
```

The script backs up an invalid/incomplete `vendor/GTR` instead of deleting it, then downloads a precise commit from GitHub. It never modifies `dataset`, `configs`, `extensions`, or `runs`.

If remote GitHub ZIP is blocked, download this archive on your **Windows** computer, then upload to the remote project root in VS Code:

https://github.com/macovaseas/GTR/archive/d94161906151bdc0c94f4d21c2b0ca356aeb3135.zip

Rename the uploaded ZIP to `GTR_official_source.zip` and run `bash 05_fix_official_download.sh` again. It will be verified for the exact pinned source version before use. If the workspace shows another error, run `df -h /root/autodl-tmp` and `df -i /root/autodl-tmp`, then share the result without any passwords.
