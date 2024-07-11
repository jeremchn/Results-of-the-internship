# Predicting TE characteristics

## Repository organization

* the `data` folder should contain the following folders (to obtain, not on github):
** `89_summaries`: 

** `89_blasted`: first database of fixed TEs, blasted and annotated with genes and methylation features.
Already filtered for pericentromeres and more than 100bp lengths, but not for closely neighboring TEs.

** `89fixedTE_blasted`: more up to date database of fixed TEs, blasted and annotated with genes and methylation features.
Already filtered for pericentromeres and more than 100bp lengths, but not for closely neighboring TEs.

* the `PericentromericV2.ipynb` notebook is used to define the pericentromeric region in each chromosome for each genome (from `89_summaries`), and filter out TEs that are
** more than 100bp in length
** less than 500bp from the nearest TE
** in the pericentromeric region

* the `output` should contain the following folders (to create):
** `89_summaries`: for the outputs created from data in `data/89_summaries`

** `89_blasted`: for the outputs created from data in `data/89_blasted`

** `89fixedTE_blasted`: for the outputs created from data in `data/89fixedTE_blasted`
