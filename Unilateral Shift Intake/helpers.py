import numpy as np

'''
################################################
# index dictionary contains all important information which is needed for modelling
# general values for country, diseases, risks, ages, and genders, as well as categories for marginal change scenarios
################################################
'''
scenario_names = np.array(['CT', 'FT']) # this is a place-holder and can be modified by the user (corresponds to scenario names in shift file, there could be multiple)

time_points = np.array(['2025', '2030']) # this is a place-holder; can have multiplie time-points to create projections into the future 

# list of countries included in the emulator
# ordering is important, as country indices are used throughout the model 
countries = np.array(['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Antigua and Barbuda',
             'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Barbados',
             'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',
             'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada',
             'Cape Verde', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo',
             'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic',
             'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador',
             'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia',
             'Federated States of Micronesia', 'Fiji', 'Finland', 'France', 'Gabon', 'Georgia', 'Germany', 'Ghana',
             'Greece', 'Greenland', 'Grenada', 'Guam', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',
             'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
             'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia',
             'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 'Malawi',
             'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico',
             'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands',
             'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Northern Mariana Islands', 'Norway',
             'Oman', 'Pakistan', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland',
             'Portugal', 'Puerto Rico', 'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 'Saint Lucia',
             'Saint Vincent and the Grenadines', 'Samoa', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
             'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia',
             'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland',
             'Sweden', 'Switzerland', 'Syria', 'Taiwan (Province of China)', 'Tajikistan', 'Tanzania', 'Thailand',
             'The Bahamas', 'The Gambia', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
             'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United States', 'Uruguay',
             'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'])

# dietary risk factors included in the GBD Diet-NCD framework 
risks = np.array(['Diet low in calcium', 'Diet low in fiber', 'Diet low in seafood omega-3 fatty acids',
                 'Diet low in fruits', 'Diet low in whole grains', 'Diet low in legumes', 'Diet low in milk',
                 'Diet low in nuts and seeds', 'Diet high in processed meat',
                 'Diet low in polyunsaturated fatty acids', 'Diet high in red meat', 'Diet high in sodium',
                 'Diet high in sugar-sweetened beverages', 'Diet high in trans fatty acids',
                 'Diet low in vegetables'])

# disease outcomes linked to dietary risks 
diseases = np.array(['Colon and rectum cancer', 'Diabetes mellitus type 2', 'Esophageal cancer', 'Intracerebral hemorrhage',
            'Ischemic heart disease', 'Ischemic stroke', 'Larynx cancer', 'Lip and oral cavity cancer',
            'Nasopharynx cancer', 'Other pharynx cancer', 'Stomach cancer', 'Subarachnoid hemorrhage',
            'Tracheal, bronchus, and lung cancer'])

# age groups considered in the model (GBD age categories)
age_groups = np.array(['25 to 29', '30 to 34', '35 to 39', '40 to 44', '45 to 49', '50 to 54', '55 to 59', '60 to 64',
                      '65 to 69', '70 to 74', '75 to 79', '80 to 84', '85 to 89', '90 to 94', '95 plus'])

# sex categories
genders = ['Female', 'Male']

# central index dictionary used across all scripts for this analytical scenario  
# provides consistent indexing for nested loops and data alignment  
index_dict = {'scenario_names': scenario_names, 'time_points':time_points, 'countries': countries, 'risks': risks, 'age_groups': age_groups, 'diseases': diseases,
              'genders': genders}

# tuples specifying distribution parameter names
dist_parameter_tuples = [('expon', 'scale'), ('gamma', 'scale'), ('gamma', 'a'), ('fisk', 'scale'), ('fisk', 'c'),
                            ('gumbel_r', 'scale'), ('gumbel_r', 'loc'), ('weibull_min', 'scale'),
                            ('weibull_min', 'c'), ('lognorm', 'scale'), ('lognorm', 's'),  ('norm', 'scale'),
                            ('norm', 'loc'), ('beta', 'scale'), ('beta', 'loc'), ('beta', 'a'), ('beta', 'b'),
                            ('mirrored_gamma', 'scale'), ('mirrored_gamma', 'a'), ('mirrored_gumbel_r', 'scale'),
                            ('mirrored_gumbel_r', 'loc'), ('invgamma', 'scale'), ('invgamma', 'a'),
                            ('invweibull', 'scale'), ('invweibull', 'c'), ('limits', 'lower'), ('limits', 'upper')]

# list of probability distribution families supported by the emulator
distribution_names = ['expon', 'gamma', 'fisk', 'gumbel_r', 'weibull_min', 'lognorm', 'norm', 'beta', 'mirrored_gamma',
                      'mirrored_gumbel_r', 'invgamma', 'invweibull']









