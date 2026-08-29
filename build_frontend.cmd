@echo off
setlocal enabledelayedexpansion

rem Windows counterpart of build_frontend.sh.
rem
rem The frontend used to be built on macOS only, which is why the export is
rem committed to the repository at all -- a Windows machine could not produce
rem one. It can: pnpm comes with Node via corepack, and the Next.js export needs
rem nothing else.
rem
rem Same order as the shell version: build first, swap afterwards, so a failed
rem build leaves the packaged copy alone rather than deleting the UI.

rem corepack otherwise stops to ask before fetching the pinned pnpm.
set COREPACK_ENABLE_DOWNLOAD_PROMPT=0

cd /d "%~dp0frontend" || exit /b 1

call corepack pnpm install
if errorlevel 1 (
    echo pnpm install failed 1>&2
    exit /b 1
)

call corepack pnpm run build
if errorlevel 1 (
    echo frontend build failed 1>&2
    exit /b 1
)

cd /d "%~dp0"

if not exist "frontend\out" (
    echo frontend build produced no output directory ^(frontend\out^) 1>&2
    exit /b 1
)

if exist "anylearning\frontend-dist" rmdir /s /q "anylearning\frontend-dist"
move "frontend\out" "anylearning\frontend-dist" >nul
if errorlevel 1 (
    echo could not move frontend\out into anylearning\frontend-dist 1>&2
    exit /b 1
)
type nul > "anylearning\frontend-dist\__init__.py"

echo Frontend built into anylearning\frontend-dist
