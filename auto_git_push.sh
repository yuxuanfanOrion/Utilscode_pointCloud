#!/bin/bash
# fyx - 2024-05-04
'''
usage: ./auto_git_push.sh "commit message"
example: 
  ./auto_git_push.sh "README.md"
  github will show:  update README.md - 2024-04-27 11:12:44
'''

# Scripts begining here
# check if user provides commit message
if [ -z "$1" ]; then
  echo "Please provide a commit message meow meow。"
  exit 1
fi

# get current date and time
current_date=$(date "+%Y-%m-%d %H:%M:%S")

# the message user provides
commit_message="$1"

# auto add all changes
git add .

# commit with the message user provides and current date
git commit -m "update $commit_message - $current_date"

# 推送到远程仓库的默认分支
git push