@echo off
chcp 65001 >nul
echo ==============================================
echo   一键启动机器学习环境 (Python 3.9)
echo   by ChatGPT
echo ==============================================
cd /d E:\smartknee\Smart-Knee-Guard\data-analysis

:: 解除 PowerShell 执行策略限制（当前用户）
powershell -Command "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"

:: 激活虚拟环境
call ml_env\Scripts\activate

:: 显示提示信息
echo.
echo ✅ 虚拟环境已激活！
echo   现在你可以运行 python、pip 或 jupyter 等命令。
echo.

:: 可选：自动进入 Python 解释器（按 Ctrl+Z + 回车 退出）
python
