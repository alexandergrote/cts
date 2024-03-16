@echo off

REM my/path/multi_file.bat synthetic

for %%f in ("boruta", "mutinfo", "rf", "cts") do (

    start bat\main\single_file %1 %%f

)