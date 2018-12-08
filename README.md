# Malaria outbreaks prediction

![network architecure](https://github.com/mustafaghali/malaria_outbreaks_prediction/blob/master/modeling/Net.pdf)

A geo-coded inventory of anophelines has been modelled (spatio-temporally)
to predict malaria outbreaks in Africa. After pre-processing the data, the
predictions tasks were diferent in terms of target feature, such as predicting
specie types in which we used several methods for training and testing proce-
dures including binary classifcation for individual species and the multilabel
classifcation and compared the performance of those models to select the
most accurate one for our work. We also tried predictions on the location
of the next outbreak given the time and species types with two diferent ap-
proaches, Latitude and longitude regression task and country classifcation
using Neural Networks, from the diferent results obtained we can say that
the data we tried to model was having very high noise supported by what
we have shown in the exploratory analysis section, which indicates that more
refning on the data should be done (from the sources) as well as verifying
the underling assumptions of the data understanding.
