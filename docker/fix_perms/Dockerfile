# Generic permission fixer: runs cron, re-applying world-readable bits
# (a+rX) to every directory in TARGET_DIRS on a configurable schedule.
# Use it for any container-managed bind mount where the host user
# should keep read access.
#
# In compose, set TARGET_DIRS to a space-separated list of mount paths
# (e.g. "/data /artifacts") and CRON_SCHEDULE to a standard cron
# expression (default: every minute).
FROM alpine:3

ENV TARGET_DIRS=/data
ENV CRON_SCHEDULE="*/1 * * * *"

# The fixer script reads TARGET_DIRS from its environment.
RUN printf '%s\n' \
      '#!/bin/sh' \
      'for d in $TARGET_DIRS; do' \
      '  chmod -R a+rX "$d"' \
      'done' \
      > /usr/local/bin/fix-perms && chmod +x /usr/local/bin/fix-perms

# At startup: write root's crontab (cron jobs do NOT inherit the
# container env, so TARGET_DIRS is exported into the cron environment),
# run the fixer once immediately, then run cron in the foreground.
CMD ["sh", "-c", "printf 'TARGET_DIRS=\"%s\"\\n%s /usr/local/bin/fix-perms\\n' \"$TARGET_DIRS\" \"$CRON_SCHEDULE\" > /etc/crontabs/root && /usr/local/bin/fix-perms && exec crond -f"]
