# Predicting TE characteristics

## Repository organization

* the `data` folder should contain the following folders (to obtain, not on github):
  * `89_summaries`: 
  * `89_blasted`: first database of fixed TEs, blasted and annotated with genes and methylation features.
Already filtered for pericentromeres and more than 100bp lengths, but not for closely neighboring TEs.
  * `89fixedTE_blasted`: more up to date database of fixed TEs, blasted and annotated with genes and methylation features.
Already filtered for pericentromeres and more than 100bp lengths, but not for closely neighboring TEs.

* the `PericentromericV2.ipynb` notebook is used to define the pericentromeric region in each chromosome for each genome (from `89_summaries`), and filter out TEs that are
  * more than 100bp in length
  * less than 500bp from the nearest TE
  * in the pericentromeric region

* the `output` should contain the following folders (to create):
  * `89_summaries`: for the outputs created from data in `data/89_summaries`
  * `89_blasted`: for the outputs created from data in `data/89_blasted`
  * `89fixedTE_blasted`: for the outputs created from data in `data/89fixedTE_blasted`

* These 3 notebooks have the same pipeline :
  * Preprocessing : import our datas
  * Filtering our data : filtering our datas with the 3 filters : filtering short TEs, TEs that are too close to the previous TE and TEs that are in the pericentromeres regions
  * Features Meth divided by Context : We divided all the features Meth by their Context to have the the rate of methylation
  * Features Start and End transform in length : The length is a feature more coherent
  * Classification Model : Predict the superfamily of the TEs with a One VS All model.
  * Regression Model : Predict the spreading of methylation in one window thanks to the features before this window

    
    
  
