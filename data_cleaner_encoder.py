#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:00:31 2018

@author: Marianneabemgnigni, kobbypanfordquainoo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class cleandata:  
    
    def get_encoded_data(self, url):
        #self.url  = url
        data = pd.read_csv(url, sep = ',', encoding = "ISO-8859-1")
        #Selecting the features to keep
        List=['Country','Full_Name', 'Lat', 'Long','YeStart','YeEnd', 'An gambiae_complex', 'An gambiae ss', 'SS M Form (An colluzzi or Mopti forms)', 'SS S Form (savanah or Bamako forms)','An arabiensis','An. melas','An. merus','An bwambae','An funestus  s.l','An funestus s.s. (specified)','An rivulorum','An leesoni','An parensis','An vaneedeni','An nili s.l','An moucheti s.l','An pharoensis','An hancocki','An mascarensis','An marshalli','An squamous','An wellcomei','An rufipes','An coustani s.l','An ziemanni ','An paludis ','Adults/Larvae']
        New_data= data[List]
        #Taking the mean over the two years, round is to make sure we do not have decimals in years 
        mylist = list(round(New_data[['YeStart', 'YeEnd']].mean(axis=1)))
        mylist = np.array(mylist).reshape(-1,1)
        #We then replace the year start with those values
        New_data['YeStart'] = mylist
        New_data.columns = ['year' if x=='YeStart' else x for x in New_data.columns]
        New_data=New_data.drop('YeEnd',1)
        New_data['Lat'].isnull().sum()
        New_data['Long'].isnull().sum()
        New_data = New_data.dropna(axis=0, subset=['Lat'])
        encoded_data = New_data.replace(np.nan,0).replace('Y',1)
        encoded_data=encoded_data.reset_index(drop=True)
        encoded_data.rename(columns=lambda x: x.strip())
        encoded_data=encoded_data.drop('Adults/Larvae',1)
        return encoded_data

    def visualise_data(self, coded_data,species_start_column, species_end_column):
        species = coded_data.iloc[:,species_start_column:species_end_column]
        plt.figure(figsize=(12,12))
        c = 0
        for i in range(1,(len(species.columns) + 1)):    
            c = i * species.iloc[:,i - 1]
            plt.scatter(coded_data.Long, coded_data.Lat, c,linewidths=2 , marker = '.', label = species.columns[i-1])
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Geo-coded Anopheline Malaria Inventory Study')
    
    
    def visualise_country_specific(self, coded_data, species_start_column,species_end_column):
        
        for country in coded_data['Country'].unique():
    
            def get_country(country_name):
                country_data = coded_data[coded_data['Country'] == country_name]
                return country_data
    
            country_data = get_country(country)
            #country_data = country_data.dropna(axis = 1, how = 'all')
            plt.figure(figsize=(8,8))
            #plt.subplots(1,3)
    
            for i in range(species_start_column,species_end_column):  
                c = 0
                c = i * country_data.iloc[:,i]
                plt.title(country)
                plt.scatter(country_data.Long, country_data.Lat, c ,linewidths=3, marker = '.',alpha=1, label = country_data.columns[i])
    
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')  
        return
    

