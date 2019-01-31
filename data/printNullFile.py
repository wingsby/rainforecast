
import os
import shutil

path1='/dpdata/Forecast0925/'
path2='/dpdata/Forecast1016/'
path3='/dpdata/Forecast1016/'

dir1s=os.listdir(path1)
dir2s=os.listdir(path2)
dirs=list(set(dir1s)^set(dir2s))

for dir1 in dirs:
    # shutil.copytree(path1+dir1,path3+dir1)
    shutil.move(path1+dir1,path3+dir1)
