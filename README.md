# gaussnewtonmf

# Developement Guide

Every matlab code commited to this repo should pass the formatter (provided by https://marketplace.visualstudio.com/items?itemName=AffenWiesel.matlab-formatter).

## Usage

- Install MATLAB formatter.
```bash
cd formatter
./init_formatter.sh
```

- Commit a new/changed MATLAB code. For example, if we want to commit example.m.
```bash
git add example.m 
git commit                # This will call the matlab formatter. If your MATLAB code is passed, then it will proceed to commit-message page.
git diff example.m        # The MATLAB formatter will change your file accordingly, so review all not-passed files. 
git add example.m         # Re-add your files.
git commit                # If your MATLAB code is passed, then it will proceed to commit-message page.
```
