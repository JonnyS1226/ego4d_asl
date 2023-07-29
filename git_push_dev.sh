# 如果项目已经初始化后，已经init 那么不用加这个
# git init
# 更新your对应分支
git pull origin dev
# 查看本地状态
git status
# 添加本地修改的文件
git add .

# 提交
git commit -m $1
# 添加远程remote 如果项目已经remote，可以省略
# git remote add origin https://github.com/xx.git
# 推送到指定分支
git push origin dev
