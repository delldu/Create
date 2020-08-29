## Readme

### 1. Project
#### 1.1 Name
**{{ . }}**
#### 1.2 Introduce

**{{ create "author" }}** create this version (`0.0.1`) at {{ bash "date" }}.

### 2. Build & Run 

#### 2.1 Depend
`python: 3.7.7`
`pytorch: 1.5.1`
`pillow: 7.2.0` 
`torchvision: 0.6.1`

#### 2.2 Env

`export USE_ONLY_CPU=YES | NO`
Default is **NO**, use **GPU** with high priority.
`export ENABLE_APEX=YES | NO`
Default is **YES**.

#### 2.3 Train
`python train.py --help`

#### 2.4 Test
`python test.py --help`

#### 2.5 Predict
`python predict.py --help`


