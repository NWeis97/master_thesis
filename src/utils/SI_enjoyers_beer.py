import numpy as np
import pandas as pd



index = [1,2,3,4,5,6,6.5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
brewery = ['Founders','Vocation','Sierra Nevada','Svaneke','Brewdog','Brewdog',
           'To Øl','Ebeltoft Gårdbryggeri','Founders','Omnipollo','Licorne','Superfreunde',
           'Brewdog Vs. Evil Twin','To Øl','Ebeltoft Gårdbryggeri','Brewdog','Baird Stone Ishii',
           'Beer Here DK','Omnipollo','Brewdog','Brewdog','Sapporo','Founders','Kogle','To Øl',
           'Kogle','To Øl','Kogle','Vestfyen','Brewdog','Ærø','Ørbæk']
beer_name = ['All Day Vacay','Naughty & Nice','Hazy Little Thing','Sommer Hvede','Elvis Juice',
             'Hazy Jane Guava','Nisse Juice','Damn Dark VII','All Day IPA','Imperial Julmust Holiday',
             'Noël','Hell','Roaster Coaster','Chugwork Orange','Wildflower Batch 1000','Basic shake',
             'Japanese Green Tea','Brun Sovs','Levon','Silk Road','Waltz into Winter','Premium Beer',
             'Centennial','Angry toes in Lemonade','45 Days','Accidentally not a Margarita',
             'First Frontier','Golden Greedy Bastard','Willemoes Julebryg','Hoppy Christmas','Valnød',
             'Fynsk Jul']
type = ['Wheat Ale','Stout','IPA','Wheat Ale','IPA','IPA','IPA','Stout','IPA','Sour','Juleøl',
        'Lager','Nitro Strout','Sour','IPA','IPA','IPA','Juleøl','BPA','IPA','Lager','Lager',
        'IPA','Sour','IPL','IPA','IPA','Lager','Juleøl','IPA','Lager','Juleøl']
volumne = [4.6,4.5,6.7,4.6,6.5,5.0,4.6,8.5,4.7,5.9,5.8,5.2,9.0,3.4,7.4,4.7,10.1,4.8,6.5,6.5,4.5,5.0,
           7.2,2.7,5.5,2.7,7.1,2.7,7.5,6.0,6.0,4.9]
m_liters = [473,440,335,440,440,440,330,473,330,750,500,402,440,330,330,473,330,750,440,330,
            650,473,330,440,330,440,330,500,330,500,500]
price = [33,50,25,23.5,24,40,np.nan,33,33,35,60,38,45,40,33,28,40,35,70,40,np.nan,43,33,25,40,
         25,40,25,26,26,np.nan,24]
rating_untappd = [3.57,3.8,3.79,2.97,3.72,3.73,3.31,3.53,3.70,3.18,3.20,3.54,4.04,3.56,3.82,3.48,
                  3.78,2.86,3.18,3.75,2.39,3.51,2.62,3.76,2.36,3.09,3.55,3.39,2.93]

rating_weis = [3.75,2.50,2.50,1.25,3.75,4.00,3.25,3.25,3.75,2.50,2.75,2.50,3.25,3.75,3.50,4.00,
               3.00,1.75,3.00,3.00,3.00,2.25,3.75,1.00,3.25,0.75,4.25,1.00,2.75,3.50,3.0,2.25]
price_weis = [40,35,30,20,35,30,30,35,40,25,55,25,45,35,30,25,50,20,50,35,20,20,35,20,35,25,35,
              25,25,35,30,25]
rating_patrick = [3.00,2.25,2.75,2.50,2.75,3.00,2.75,2.5,3.00,3.5,2.25,3.0,0.75,3.75,2.75,3.25,
                  2.75,2.00,3.0,3.5,2.75,3.00,2.75,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                  np.nan,np.nan,np.nan]
price_patrick = [40,35,35,40,30,30,25,40,30,35,50,30,30,35,35,29,29,-10,45,35,25,30,38,30,np.nan,
                 np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
rating_carl = [3.00,2.00,2.00,2.75,3.25,3.50,3.00,2.75,3.00,2.00,1.50,2.50,1.75,3.25,3.25,4.25,
               2.25,2.00,3.75,2.75,2.75,2.5,2.25,2.00,4.00,2.00,4.00,3.00,2.25,2.75,np.nan,3.50]
price_carl = []