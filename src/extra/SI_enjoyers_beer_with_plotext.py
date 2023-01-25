"""
Just a personal side-project, with no ties to this repo at all. 
Keeping it here for fun
"""
import numpy as np
import pandas as pd
import time
import termplotlib as tpl
from pyfiglet import Figlet
import sys
import plotext as plx


if len(sys.argv)>2:
      rating = sys.argv[1]
      tf = float(1/int(sys.argv[2]))
elif len(sys.argv)>1:
      rating = sys.argv[1]
      tf = 1
else:
      rating = 'Our'
      tf = 1

index = [1,2,3,4,5,6,6.5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
brewery = ['Founders','Vocation','Sierra Nevada','Svaneke','Brewdog','Brewdog',
           'To Øl','Ebeltoft Gårdbryggeri','Founders','Omnipollo','Licorne','Superfreunde',
           'Brewdog Vs. Evil Twin','To Øl','Ebeltoft Gårdbryggeri','Brewdog','Baird Stone Ishii',
           'Beer Here DK','Omnipollo','Brewdog','Brewdog','Sapporo','Founders','Kogle','To Øl',
           'Kogle','To Øl','Kogle','Vestfyen','Brewdog','Ærø','Ørbæk']
beer_name = ['All Day Vacay','Naughty & Nice','Hazy Little Thing','Sommer Hvede','Elvis Juice',
             'Hazy Jane Guava','Nisse Juice','Damn Dark VII','All Day IPA','Imperial Julmust Holiday',
             'Noël','Hell','Roaster Coaster','Chugwork Orange','Wildflower Batch 1000','Basic shake',
             'Japanese Green Tea','Brun Sovs','Levon','Silk Road','Lost in Lychee & Lime','Premium Beer',
             'Centennial','Angry toes in Lemonade','45 Days','Accidentally not a Margarita',
             'First Frontier','Golden Greedy Bastard','Willemoes Julebryg','Hoppy Christmas','Valnød',
             'Fynsk Jul']
type = ['Wheat Ale','Stout','IPA','Wheat Ale','IPA','IPA','IPA','Stout','IPA','Sour','Juleøl',
        'Lager','Nitro Stout','Sour','IPA','IPA','IPA','Juleøl','BPA','IPA','Lager','Lager',
        'IPA','Sour','IPL','IPA','IPA','Lager','Juleøl','IPA','Lager','Juleøl']
volume = [4.6,9.0,6.7,4.6,6.5,5.0,4.6,8.5,4.7,5.9,5.8,5.2,9.0,3.4,7.4,4.7,10.1,4.8,6.5,6.5,4.5,5.0,
           7.2,2.7,5.5,2.7,7.1,2.7,7.5,6.0,6.0,4.9]
m_liters = [473,440,335,440,440,440,330,330,473,330,750,500,402,440,330,330,473,330,750,440,330,
            650,473,330,440,330,440,330,500,330,500,500]
price = [33,50,25,23.5,24,40,35,33,33,35,60,38,45,40,33,28,40,35,70,40,30,43,33,25,40,
         25,40,25,26,26,np.nan,24]
rating_untappd = np.array([3.57,3.8,3.79,2.97,3.72,3.73,3.31,3.53,3.70,3.18,3.20,3.54,4.04,3.56,3.82,3.48,
                  3.78,2.86,3.66,3.71,3.32,3.18,3.75,2.39,3.51,2.62,3.76,2.36,3.09,3.55,3.39,2.93])

rating_weis = np.array([3.75,2.50,2.50,1.25,3.75,4.00,3.0,3.25,3.75,2.50,2.75,2.50,3.25,3.75,3.50,4.00,
               3.00,1.75,3.00,3.00,3.00,2.25,3.75,1.00,3.25,0.75,4.25,1.00,2.75,3.50,3.0,2.25])
price_weis = np.array([40,35,30,20,35,30,30,35,40,25,55,25,45,35,30,25,50,20,50,35,20,20,35,20,35,25,35,
              25,25,35,30,25])
rating_patrick = np.array([3.00,2.25,2.75,2.50,2.75,3.00,2.75,2.5,3.00,3.5,2.25,3.0,0.75,3.75,2.75,3.25,
                  2.75,2.00,3.0,3.5,2.75,3.00,2.75,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                  np.nan,np.nan,np.nan])
price_patrick = np.array([40,35,35,40,30,30,25,40,30,35,50,30,30,35,35,29,29,-10,45,35,25,30,38,30,np.nan,
                 np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
rating_carl = np.array([3.00,2.00,2.00,2.75,3.25,3.50,3.00,2.75,3.00,2.00,1.50,2.50,1.75,3.25,3.25,4.25,
               2.25,2.00,3.75,2.75,2.75,2.5,2.25,2.00,4.00,2.00,4.00,3.00,2.25,2.75,np.nan,3.50])
price_carl = np.array([22,30,25,20,35,36,22,40,25,40,55,20,35,45,31,28.5,52.5,23,44,32,30,32.5,37,22,40,
              33,45,30,30,25,30,45])
rating_rasmus = np.array([3.0,3.5,2.0,0.5,3.75,4.0,3.25,2.5,3.25,3.75,2.0,1.5,4.25,4.25,3.5,2.5,2.75,1.75,
                2.75,4.25,2.75,2.0,3.0,2.25,3.25,1.5,3.25,1.75,2.25,3.25,2.5,2.25])
price_rasmus = np.array([50,40,20,20,50,45,35,25,50,30,70,20,50,50,30,30,40,30,35,40,25,15,50,30,30,20,
                35,25,35,25,30,22])
rating_peter = np.array([3.0,2.25,2.75,3.50,3.50,3.75,3.00,3.75,4.00,3.50,2.50,2.50,3.75,4.25,3.75,2.50,
                3.75,2.25,3.00,4.00,3.25,2.75,3.75,2.50,3.25,1.75,3.75,2.00,2.25,3.5,3.0,3.25])
price_peter = np.array([25,35,25,30,35,35,25,30,35,40,60,25,40,35,30.02,30,40,45,50,40,22,35,35,20,30,20,
               35,20,45,22,30,45])

df = pd.DataFrame({'Beer Name':beer_name,'Type':type,'Brewery':brewery,'Vol (%)':volume,
                   'Price (kr)':price, 'Size (mL)':m_liters, 'Untappd Rat.':rating_untappd},
                  index=index)
df.index.name = 'Day'
df['Our'] = np.nanmean([rating_weis,rating_peter,rating_patrick,rating_rasmus,rating_carl],axis=0)
df['Price (kr/L)'] = df['Price (kr)']/(df['Size (mL)']/1000)
df['Worth (%/Kr)'] = df['Vol (%)']/df['Price (kr/L)']
df['Worth (rating)'] = df['Untappd Rat.']/df['Price (kr/L)']
df['Worth (rating us)'] = df['Our']/df['Price (kr/L)']
df = df.drop(columns=['Price (kr)'])

print("\n\n\n")
print("Loading")
time.sleep(0.5*tf)
print(".")
time.sleep(0.5*tf)
print("..")
time.sleep(0.5*tf)
print("...")
time.sleep(0.5*tf)
print("....")
time.sleep(0.5*tf)
print(".....\n\n")
time.sleep(0.5*tf)

print("\t\t*********************************************************\n")
print("\t\t***** SI Enjoyers Yearly Beer Tasting Review (2022) *****\n")
print("\t\t*********************************************************\n\n\n")
time.sleep(4*tf)

print("Hello there! And welcome to SI Enjoyers yearly beer tasting review (2022).\n")
time.sleep(5*tf)
print(f"This year, we had the wonderful experience of tasting nothing less but {len(df)} beers"+
      f" from {len(df['Brewery'].unique())} different breweries.\n\n")

input('>')

print("Below is the distribution of breweries tasted:\n----------------------------")
print(df['Brewery'].value_counts())
print("----------------------------\n")
input('>')
print(f"A total of "+
      f"{len(df['Type'].unique())} styles were consumed. The most common was "+
      f"'{df['Type'].value_counts().index[df['Type'].value_counts().argmax()]}' with "+
      f"{df['Type'].value_counts().max()} different beers.\n")
input('>')
print("Below is the distribution of types tasted:\n---------------------")
print(df['Type'].value_counts())
print("---------------------\n\n")
input('>')


print("\n\n#####################")
print("####### Price #######")
print("#####################\n")
print("We tried a varity of both cheap and more expensive beers. Below is the top five least"+
      " expensive beers (in kr/L):")
print("------------------------------------------------------------------------------------------------------")
print(df.sort_values('Price (kr/L)').head(5).iloc[:,:-3])
print("------------------------------------------------------------------------------------------------------\n")
input('>')

print("\n... and the five most"+
      " expensive beers (in kr/L):")
print("--------------------------------------------------------------------------------------------------------------------------")
print(df.sort_values('Price (kr/L)',ascending=False).head(5).iloc[:,:-3])
print("--------------------------------------------------------------------------------------------------------------------------\n")
input('>')


print("\n\n######################")
print("####### Volume #######")
print("######################\n")
print("It was no wonder we all got a bit tipsy... These were the five strongest beer we had:")
print("--------------------------------------------------------------------------------------------------------------------------")
print(df.sort_values('Vol (%)',ascending=False).head(5).iloc[:,:-3])
print("--------------------------------------------------------------------------------------------------------------------------\n")
input('>')

print("\n... and a few watery ones to rinse the throat:")
print("------------------------------------------------------------------------------------------------------")
print(df.sort_values('Vol (%)',ascending=True).head(5).iloc[:,:-3])
print("------------------------------------------------------------------------------------------------------\n")
input('>')



print("\n\n################################")
print("####### Rating (Untappd) #######")
print("################################\n")
print("Some beers were good, and some were... less good. Below we have the five highest ranked"+
      " beers based on the\nawarding winning global rating system 'Untappd':")
print("--------------------------------------------------------------------------------------------------------------------------")
print(df.sort_values('Untappd Rat.',ascending=False).head(5).iloc[:,:-3])
print("--------------------------------------------------------------------------------------------------------------------------\n")
input('>')

print("\n... and to the common people, the five worst beers: (Do you see the pattern?)")
print("-------------------------------------------------------------------------------------------------------------------")
print(df.sort_values('Untappd Rat.',ascending=True).head(5).iloc[:,:-3])
print("-------------------------------------------------------------------------------------------------------------------\n")
input('>')

"""
print("\n\n##########################")
print("####### Our Rating #######")
print("##########################\n")
print("But how interesting is the rating of the common people really...? Some people out there"+
      " give Tuborg Grøn 5/5, and\nthey can for that reason of course not be trusted. \nMuch more"+
      " interesting is the average rating among us... Here are our top five beers:")
print("--------------------------------------------------------------------------------------------------------------------------")
print(df.sort_values('Our',ascending=False).head(5).iloc[:,:-3])
print("--------------------------------------------------------------------------------------------------------------------------\n")
input('>')

print("\n... and the worst five beers:")
print("-------------------------------------------------------------------------------------------------------------------")
print(df.sort_values('Our',ascending=True).head(5).iloc[:,:-3])
print("-------------------------------------------------------------------------------------------------------------------\n")
input('>')
"""

print("\n\n################################################################################")
print("####### Worth... in terms of price per vol (The 'Kapers Hockel Measure') #######")
print("################################################################################\n")
print("Do I really have to clarify this one...?")
print("--------------------------------------------------------------------------------------------------------------------------")
print(df.sort_values('Worth (%/Kr)',ascending=False).head(5).iloc[:,:-2])
print("--------------------------------------------------------------------------------------------------------------------------\n")
input('>')

print("\n... and to the intelligent alcoholic, the worst five beers (how precarious the top bottom"+
      " brewery is 'Kogle' once again...):")
print("-------------------------------------------------------------------------------------------------------------------------------")
print(df.sort_values('Worth (%/Kr)',ascending=True).head(5).iloc[:,:-2])
print("-------------------------------------------------------------------------------------------------------------------------------\n")
input('>')


print("\n\n#############################################################")
print("####### Worth... in terms of price per Untappd rating #######")
print("#############################################################\n")
print("The five most worth beers to the common people")
print("-----------------------------------------------------------------------------------------------------------------------")
print(df.sort_values('Worth (rating)',ascending=False).head(5).drop(columns=['Worth (%/Kr)','Worth (rating us)']))
print("-----------------------------------------------------------------------------------------------------------------------\n")
input('>')

print("\n... the five least worth beers come to no surprose")
print("------------------------------------------------------------------------------------------------------------------------------")
print(df.sort_values('Worth (rating)',ascending=True).head(5).drop(columns=['Worth (%/Kr)','Worth (rating us)']))
print("------------------------------------------------------------------------------------------------------------------------------\n")
input('>')

print(".")
print(".")
print(".")
print(".")
print(".")
print(".")

df['Weis'] = np.nan_to_num(rating_weis,nan=2.5)
df['Peter'] = np.nan_to_num(rating_peter,nan=2.5)
df['Patrick'] = np.nan_to_num(rating_patrick,nan=2.5)
df['Carl'] = np.nan_to_num(rating_carl,nan=2.5)
df['Rasmus'] = np.nan_to_num(rating_rasmus,nan=2.5)


print("\n")
print("Enough with common stats... let's dig a bit deeper into some interesting stats based on our"+
      " voting:")
time.sleep(0.5*tf)
print(".")
time.sleep(0.5*tf)
print("..")
time.sleep(0.5*tf)
print("...")
time.sleep(0.5*tf)
print("....")
time.sleep(0.5*tf)
print(".....\n\n")
time.sleep(0.5*tf)

str_=f"================================ STATS FOR {rating.upper()} RATINGS =================================="
print(str("\n\t"+"="*len(str_)))
print("\t"+str_)
print(str("\t"+"="*len(str_)))
print("\n")
print("\nNB: Text for each graph is only meant for 'Our Rating', not individual.")

input('>')



print("________________________________________________________________________________________________________________________\n")
str_ = f"####### Type vs. {rating} Rating #######"
print(str("\n\t\t"+"#"*len(str_)))
print("\t\t"+str_)
print(str("\t\t"+"#"*len(str_)))
print("\n")
fig = tpl.figure()
type_ = df.groupby(['Type']).mean().sort_values([rating],ascending=False).loc[:,[rating]]
fig.barh(np.round((type_.values),3).squeeze().tolist(), type_.index, force_ascii=False)
fig.show()
input('>')
print("________________________________________________________________________________________________________________________\n")

str_=f"####### Brewery vs. {rating} Rating #######"
print(str("\n\t\t"+"#"*len(str_)))
print("\t\t"+str_)
print(str("\t\t"+"#"*len(str_)))
print("\n")
fig = tpl.figure()
type_ = df.groupby(['Brewery']).mean().sort_values([rating],ascending=False).loc[:,[rating]]
fig.barh(np.round((type_.values),3).squeeze().tolist(), type_.index, force_ascii=False)
fig.show()
input('>')
print("________________________________________________________________________________________________________________________\n")

rat_ticks = [1*i for i in range(6)]
rat_labels = [str(i)+'.00'  for i in rat_ticks]
      
str_=f"####### Volume vs. {rating} Rating #######"
print(str("\n\t"+"#"*len(str_)))
print("\t"+str_)
print(str("\t"+"#"*len(str_)))
print("\nHere we see a upward going trend for rating, when the volume is increased.")
print("But when the volume becomes to high, the rating settles at around 3.")
x = df['Vol (%)'].fillna(6)
y = df[rating]
plx.scatter(x,y)
plx.plotsize(60,22)
plx.xlabel("Volume (%)")
plx.title(f"Volume vs. '{rating}' Rating")
plx.ylim(0,5)
plx.xlim(0,12)
plx.yticks(rat_ticks,rat_labels)
plx.show()
input('>')
print("________________________________________________________________________________________________________________________\n")



str_=f"####### Untappd Rating vs. {rating} Rating #######"
print(str("\n\t"+"#"*len(str_)))
print("\t"+str_)
print(str("\t"+"#"*len(str_)))
print("\nWe seem to agree quite well with the common people.")
x = df['Untappd Rat.'].fillna(2.5)
y = df[rating]
plx.clear_figure()
plx.scatter(x,y)
plx.plotsize(60,22)
plx.xlabel("Untappd Rating")
plx.title(f"Untappd Rating vs. '{rating}' Rating")
plx.ylim(0,5)
plx.xlim(0,5)
plx.xticks(rat_ticks,rat_labels)
plx.yticks(rat_ticks,rat_labels)
plx.show()
input('>')
print("________________________________________________________________________________________________________________________\n")



str_=f"####### Size vs. {rating} Rating #######"
print(str("\n\t"+"#"*len(str_)))
print("\t"+str_)
print(str("\t"+"#"*len(str_)))
print("\nThe closer to 440 mL the better!")
x = df['Size (mL)'].fillna(400)
y = df[rating]
plx.clear_figure()
plx.scatter(x,y)
plx.plotsize(60,22)
plx.xlabel("Size (mL)")
plx.title(f"Size (mL) vs. '{rating}' Rating")
plx.ylim(0,5)
plx.xlim(200,800)
plx.yticks(rat_ticks,rat_labels)
plx.show()
input('>')
print("________________________________________________________________________________________________________________________\n")



str_=f"####### Price per Liter vs. {rating} Rating #######"
print(str("\n\t"+"#"*len(str_)))
print("\t"+str_)
print(str("\t"+"#"*len(str_)))
print("\nThere seem to be no clear indication that more expensive beers are better...")
x = df['Price (kr/L)'].fillna(80)
y = df[rating].values
plx.clear_figure()
plx.scatter(x,y)
plx.plotsize(60,22)
plx.xlabel("Price (kr/L)")
plx.title(f"Price (kr/L) vs. '{rating}' Rating")
plx.ylim(0,5)
plx.xlim(0,120)
plx.yticks(rat_ticks,rat_labels)
plx.show()
input('>')
print("________________________________________________________________________________________________________________________\n")



str_=f"####### Time (day) vs. {rating} Rating #######"
print(str("\n\t"+"#"*len(str_)))
print("\t"+str_)
print(str("\t"+"#"*len(str_)))
print("\nLooks at bit like our heartbeat after 30 beers xD")
x = df.index
y = df[rating].values
plx.clear_figure()
plx.plot(x,y)
plx.plotsize(60,22)
plx.xlabel("Day")
plx.title(f"Day vs. '{rating}' Rating")
plx.ylim(0,5)
plx.xlim(0,32)
plx.yticks(rat_ticks,rat_labels)
plx.show()
input('>')
print("________________________________________________________________________________________________________________________\n")



print("\n\n________________________________________________________________________________________________________________________")
print("________________________________________________________________________________________________________________________")
print("________________________________________________________________________________________________________________________")
print(f"\nIt's time to some intern comparison between us... First lets have a look at {rating}'s"+
      " ratings vs. all five of us.\n\n")
input('>')
x = df[rating]
plx.clear_figure()
plx.scatter(x,df['Weis'])
plx.plotsize(60,22)
plx.xlabel(f"'{rating}' Rating")
plx.title(f"'{rating}' Rating vs. 'Weis' Rating")
plx.ylim(0,5)
plx.xlim(0,5)
plx.xticks(rat_ticks,rat_labels)
plx.yticks(rat_ticks,rat_labels)
plx.show()
print("\n\n")
input('>')
plx.clear_figure()
plx.scatter(x,df['Peter'])
plx.plotsize(60,22)
plx.xlabel(f"'{rating}' Rating")
plx.title(f"'{rating}' Rating vs. 'Peter' Rating")
plx.ylim(0,5)
plx.xlim(0,5)
plx.xticks(rat_ticks,rat_labels)
plx.yticks(rat_ticks,rat_labels)
plx.show()
print("\n\n")
input('>')
plx.clear_figure()
plx.scatter(x,df['Carl'])
plx.plotsize(60,22)
plx.xlabel(f"'{rating}' Rating")
plx.title(f"'{rating}' Rating vs. 'Carl' Rating")
plx.ylim(0,5)
plx.xlim(0,5)
plx.xticks(rat_ticks,rat_labels)
plx.yticks(rat_ticks,rat_labels)
plx.show()
print("\n\n")
input('>')
plx.clear_figure()
plx.scatter(x,df['Patrick'])
plx.plotsize(60,22)
plx.xlabel(f"'{rating}' Rating")
plx.title(f"'{rating}' Rating vs. 'Patrick' Rating")
plx.ylim(0,5)
plx.xlim(0,5)
plx.xticks(rat_ticks,rat_labels)
plx.yticks(rat_ticks,rat_labels)
plx.show()
print("\n\n")
input('>')
plx.clear_figure()
plx.scatter(x,df['Rasmus'])
plx.plotsize(60,22)
plx.xlabel(f"'{rating}' Rating")
plx.title(f"'{rating}' Rating vs. 'Rasmus' Rating")
plx.ylim(0,5)
plx.xlim(0,5)
plx.xticks(rat_ticks,rat_labels)
plx.yticks(rat_ticks,rat_labels)
plx.show()
print("\n\n")
input('>')


print("Based on correlation between ratings, the following are the most compatible in" +
      " terms of taste in beer <3\n")
corr = np.corrcoef(df.iloc[:,-5:].values.T)
i=0
str_=f"--- Weis ---"
print(str("\n"+"-"*len(str_)))
print(""+str_)
print(str(""+"-"*len(str_)))
idx = (np.argsort(corr[i,:])[::-1]).astype(int)
df_weis_sim = pd.DataFrame({'Name':df.iloc[:,-5:].columns[idx[1:]],'Correlation':corr[i,idx[1:]]})
print(df_weis_sim)
print("\n")
input('>')

i=1
str_=f"--- Peter ---"
print(str("\n"+"-"*len(str_)))
print(""+str_)
print(str(""+"-"*len(str_)))
idx = (np.argsort(corr[i,:])[::-1]).astype(int)
df_weis_sim = pd.DataFrame({'Name':df.iloc[:,-5:].columns[idx[1:]],'Correlation':corr[i,idx[1:]]})
print(df_weis_sim)
print("\n")
input('>')

i=2
str_=f"--- Patrick ---"
print(str("\n"+"-"*len(str_)))
print(""+str_)
print(str(""+"-"*len(str_)))
idx = (np.argsort(corr[i,:])[::-1]).astype(int)
df_weis_sim = pd.DataFrame({'Name':df.iloc[:,-5:].columns[idx[1:]],'Correlation':corr[i,idx[1:]]})
print(df_weis_sim)
print("\n")
input('>')

i=3
str_=f"--- Carl ---"
print(str("\n"+"-"*len(str_)))
print(""+str_)
print(str(""+"-"*len(str_)))
idx = (np.argsort(corr[i,:])[::-1]).astype(int)
df_weis_sim = pd.DataFrame({'Name':df.iloc[:,-5:].columns[idx[1:]],'Correlation':corr[i,idx[1:]]})
print(df_weis_sim)
print("\n")
input('>')

i=4
str_=f"--- Rasmus ---"
print(str("\n"+"-"*len(str_)))
print(""+str_)
print(str(""+"-"*len(str_)))
idx = (np.argsort(corr[i,:])[::-1]).astype(int)
df_weis_sim = pd.DataFrame({'Name':df.iloc[:,-5:].columns[idx[1:]],'Correlation':corr[i,idx[1:]]})
print(df_weis_sim)
print("\n")
input('>')


print(f"\nThe time has come to a final few nominations!!!\n\n")
time.sleep(0.5*tf)
print(".")
time.sleep(0.5*tf)
print("..")
time.sleep(0.5*tf)
print("...")
time.sleep(0.5*tf)
print("....")
time.sleep(0.5*tf)
print(".....\n\n")
time.sleep(0.5*tf)

str_=f"================================ Nominations (according to {rating} Ratings) =================================="
print(str("\n\t"+"="*len(str_)))
print("\t"+str_)
print(str("\t"+"="*len(str_)))
print("\n")
input('>')

print("\n\n#############################")
print("####### Beers (worst) #######")
print("#############################\n")
time.sleep(0.5*tf)
print("The third worst beer was")
time.sleep(1.5*tf)
time.sleep(0.5*tf)
print(".")
time.sleep(0.5*tf)
print("..")
time.sleep(0.5*tf)
print("...")
time.sleep(0.5*tf)
df_beer3 = df.sort_values(rating,ascending=False).iloc[[-3],:].drop(columns=['Size (mL)','Worth (rating)','Worth (rating us)','Worth (%/Kr)'])

str_=f"**** 3: '{df_beer3['Beer Name'].values[0]}' ****"
print(str("\n"+"*"*len(str_)))
print(""+str_)
print(str(""+"*"*len(str_)))
print("--------------------------------------------------------------------------------------------------------------------------")
print(df_beer3)
print("--------------------------------------------------------------------------------------------------------------------------")
input('>')

print("\n\n\n")
print("The second worst beer was")
time.sleep(1.5*tf)
time.sleep(0.5*tf)
print(".")
time.sleep(0.5*tf)
print("..")
time.sleep(0.5*tf)
print("...")
time.sleep(0.5*tf)

df_beer2 = df.sort_values(rating,ascending=False).iloc[[-2],:].drop(columns=['Size (mL)','Worth (rating)','Worth (rating us)','Worth (%/Kr)'])

str_=f"**** 2: '{df_beer2['Beer Name'].values[0]}' ****"
print(str("\n"+"*"*len(str_)))
print(""+str_)
print(str(""+"*"*len(str_)))
print("-------------------------------------------------------------------------------------------------------------------------------")
print(df_beer2)
print("-------------------------------------------------------------------------------------------------------------------------------")

input('>')

print("\n\n\n")
print("Aaaand the worst was...")
time.sleep(1.5*tf)
time.sleep(0.5*tf)
print(".")
time.sleep(0.5*tf)
print("..")
time.sleep(0.5*tf)
print("...")
time.sleep(0.5*tf)
print("....")
time.sleep(0.5*tf)
print(".....")
time.sleep(0.5*tf)

df_beer1 = df.sort_values(rating,ascending=False).iloc[[-1],:].drop(columns=['Size (mL)','Worth (rating)','Worth (rating us)','Worth (%/Kr)'])

str_=f"**** 1: '{df_beer1['Beer Name'].values[0]}' ****"
print(str("\n"+"*"*len(str_)))
print(""+str_)
print(str(""+"*"*len(str_)))
print("-------------------------------------------------------------------------------------------------------------------------------")
print(df_beer1)
print("-------------------------------------------------------------------------------------------------------------------------------")
input('>')

print("\n\n\n\n\n")

print("\n\n############################")
print("####### Beers (best) #######")
print("############################\n")
time.sleep(1*tf)
print("The third best beer was")
time.sleep(1.5*tf)
time.sleep(0.5*tf)
print(".")
time.sleep(0.5*tf)
print("..")
time.sleep(0.5*tf)
print("...")
time.sleep(0.5*tf)
df_beer3 = df.sort_values(rating,ascending=False).head(3).iloc[[2],:].drop(columns=['Size (mL)','Worth (rating)','Worth (rating us)','Worth (%/Kr)'])

str_=f"**** 3: '{df_beer3['Beer Name'].values[0]}' ****"
print(str("\n"+"*"*len(str_)))
print(""+str_)
print(str(""+"*"*len(str_)))
print("--------------------------------------------------------------------------------------------------------------------------")
print(df_beer3)
print("--------------------------------------------------------------------------------------------------------------------------")
input('>')

print("\n\n\n")

print("The second best beer was")
time.sleep(1.5*tf)
time.sleep(0.5*tf)
print(".")
time.sleep(0.5*tf)
print("..")
time.sleep(0.5*tf)
print("...")
time.sleep(0.5*tf)

df_beer2 = df.sort_values(rating,ascending=False).head(3).iloc[[1],:].drop(columns=['Size (mL)','Worth (rating)','Worth (rating us)','Worth (%/Kr)'])

str_=f"**** 2: '{df_beer2['Beer Name'].values[0]}' ****"
print(str("\n"+"*"*len(str_)))
print(""+str_)
print(str(""+"*"*len(str_)))
print("--------------------------------------------------------------------------------------------------------------------------")
print(df_beer2)
print("--------------------------------------------------------------------------------------------------------------------------")
input('>')

print("\n\n\n")
print("Aaaand the winner was...")
time.sleep(1.5*tf)
time.sleep(0.5*tf)
print(".")
time.sleep(0.5*tf)
print("..")
time.sleep(0.5*tf)
print("...")
time.sleep(0.5*tf)
print("....")
time.sleep(0.5*tf)
print(".....")
time.sleep(0.5*tf)

df_beer1 = df.sort_values(rating,ascending=False).head(3).iloc[[0],:].drop(columns=['Size (mL)','Worth (rating)','Worth (rating us)','Worth (%/Kr)'])

str_=f"**** 1: '{df_beer1['Beer Name'].values[0]}' ****"
print(str("\n"+"*"*len(str_)))
print(""+str_)
print(str(""+"*"*len(str_)))
print("--------------------------------------------------------------------------------------------------------------------------")
print(df_beer1)
print("--------------------------------------------------------------------------------------------------------------------------")



input('>')


print("\n\n\n\n\n")
print("The Elon Musk Price goes to the beer dividing the waters between us the most:")
time.sleep(5*tf)
time.sleep(1*tf)
print(".")
time.sleep(1*tf)
print("..")
time.sleep(1*tf)
print("...")
time.sleep(1*tf)


elon_musk = df.iloc[:,-5:].values.std(axis=1).argmax()
str_=f"**** The Elon Musk Price: '{df.iloc[[elon_musk],:]['Beer Name'].values[0]}' ****"
print(str("\n"+"*"*len(str_)))
print(""+str_)
print(str(""+"*"*len(str_)))
print("----------------------------------------------------------------------------------------------------------------------------------------")
print(df.iloc[[elon_musk],:].drop(columns=['Size (mL)','Worth (rating)','Worth (rating us)','Worth (%/Kr)']))
print("----------------------------------------------------------------------------------------------------------------------------------------")
input('>')

print("\n\n\n\n\n")
print("The Basic Bitch Price goes to the beer being the most basic (rating between 2.75 and 3.25 "+
      "and lowest deviance between our ratings):")
time.sleep(4*tf)
time.sleep(1.5*tf)
print(".")
time.sleep(1.5*tf)
print("..")
time.sleep(1.5*tf)
print("...")
time.sleep(1.5*tf)
mean_between_2_3 = (df.iloc[:,-5:].values.mean(axis=1)>=2.75) &  (df.iloc[:,-5:].values.mean(axis=1)<=3.25) 
idx_basic_bitch = df.iloc[mean_between_2_3,-5:].values.std(axis=1).argmin()
basic_bitch = np.arange(0,len(mean_between_2_3),1)[mean_between_2_3][idx_basic_bitch]
str_=f"**** The Basic Bitch Price: '{df.iloc[[basic_bitch],:]['Beer Name'].values[0]}' ****"
print(str("\n"+"*"*len(str_)))
print(""+str_)
print(str(""+"*"*len(str_)))
print("-----------------------------------------------------------------------------------------------------------")
print(df.iloc[[basic_bitch],:].drop(columns=['Size (mL)','Worth (rating)','Worth (rating us)','Worth (%/Kr)']))
print("-----------------------------------------------------------------------------------------------------------")
input('>')
print("\n\n\n\n\n")

print("The Common Guy is Crazy Price goes to the beer with the highest deviance between our and"+
      " Untappd rating:")
time.sleep(4*tf)
time.sleep(1.5*tf)
print(".")
time.sleep(1.5*tf)
print("..")
time.sleep(1.5*tf)
print("...")
time.sleep(1.5*tf)
diff_our_untappd = np.abs(df['Our']-df['Untappd Rat.'])
common_guy_is_crazy = diff_our_untappd.argmax()
str_=f"**** The Common Guy is Crazy Price: '{df.iloc[[common_guy_is_crazy],:]['Beer Name'].values[0]}' ****"
print(str("\n"+"*"*len(str_)))
print(""+str_)
print(str(""+"*"*len(str_)))
print("--------------------------------------------------------------------------------------------------------------------------")
print(df.iloc[[common_guy_is_crazy],:].drop(columns=['Size (mL)','Worth (rating)','Worth (rating us)','Worth (%/Kr)']))
print("--------------------------------------------------------------------------------------------------------------------------")
input('>')
print("\n\n\n\n\n")



str_=f"============================ Winner of the Price Guessing competition =============================="
print(str("\n\t"+"="*len(str_)))
print("\t"+str_)
print(str("\t"+"="*len(str_)))
print("\n")

print("Finally! What we all have been waiting for. Who won the price guessing competition?")
input('Press ENTER to reveal')

weis_price_diff = np.sum(np.abs(np.nan_to_num(price,nan=30)-np.nan_to_num(price_weis,nan=20)))
peter_price_diff = np.sum(np.abs(np.nan_to_num(price,nan=30)-np.nan_to_num(price_peter,nan=20)))
carl_price_diff = np.sum(np.abs(np.nan_to_num(price,nan=30)-np.nan_to_num(price_carl,nan=20)))
patrick_price_diff = np.sum(np.abs(np.nan_to_num(price,nan=30)-np.nan_to_num(price_patrick,nan=20)))
rasmus_price_diff = np.sum(np.abs(np.nan_to_num(price,nan=30)-np.nan_to_num(price_rasmus,nan=20)))

price_diff = pd.DataFrame({'Name':['Weis','Peter','Carl','Patrick','Rasmus'],
                           'Price_abs_difference': [weis_price_diff,
                                                    peter_price_diff,
                                                    carl_price_diff,
                                                    patrick_price_diff,
                                                    rasmus_price_diff]})
print("Aaaand the winner was...")
time.sleep(2*tf)
time.sleep(0.5*tf)
print(".")
time.sleep(0.5*tf)
print("..")
time.sleep(0.5*tf)
print("...")
time.sleep(0.5*tf)
print("....")
time.sleep(0.5*tf)
print(".....")
time.sleep(0.5*tf)
print("......")
time.sleep(0.5*tf)
print(".......")
time.sleep(0.5*tf)
print("........")
time.sleep(0.5*tf)

custom_fig = Figlet(font='standard')
print(custom_fig.renderText("PETER !!!"))
time.sleep(1.0*tf)
print("By a mere 0.02 kr. (apparently guessing 30.02kr. on a beer price rather than 30kr. can win you the competition!)")
time.sleep(7.0*tf)
print("\nHere is the overall result:")

print("---------------------------------")
print(price_diff.sort_values(['Price_abs_difference']))
print("---------------------------------")


print("\n\nThanks for following along! I'll see you again next year <3\n\n"+
      "Upon exiting program, an excel sheet with the corresponding data will be saved to your current"+
      " location:))")
input('>')


df_all = pd.concat([df,pd.Series(price_weis,name='Weis_Price_Guess',index=index),
                    pd.Series(price_peter,name='Peter_Price_Guess',index=index),
                    pd.Series(price_carl,name='Carl_Price_Guess',index=index),
                    pd.Series(price_patrick,name='Patrick_Price_Guess',index=index),
                    pd.Series(price_rasmus,name='Rasmus_Price_Guess',index=index)],axis=1)

df_all.to_excel('SI_enjoyers_beer_review_2022.xlsx')



        