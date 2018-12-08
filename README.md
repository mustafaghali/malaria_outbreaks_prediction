# Malaria outbreaks prediction


A geo-coded inventory of anophelines has been modelled (spatio-temporally)
to predict malaria outbreaks in Africa. After pre-processing the data, the
predictions tasks were diferent in terms of target feature, such as predicting
specie types in which we used several methods for training and testing procedures including binary classifcation for individual species and the multilabel
classifcation and compared the performance of those models to select the
most accurate one for our work. <br /> 
We also tried predictions on the location of the possible next outbreak given the time and species types with two diferent approaches:<br /> 
Latitude and longitude regression task and countrybased classifcation
using MLP with residual connections using Pytorch framework.<br /> 
![network architecure](https://github.com/mustafaghali/malaria_outbreaks_prediction/blob/master/modeling/Net-Dark.png)
