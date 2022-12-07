library(pacman)

p_load(data.table)
p_load(tidyverse)
p_load(ggplot2)
p_load(readxl)
p_load(DataExplorer)
p_load(janitor)
p_load(naniar)
p_load(vip)
p_load(tidymodels)
p_load(e1071)
p_load(esquisse)
p_load(robustbase)
p_load(outliers)
p_load(plotly)
p_load(corrplot)
p_load(Boruta)
p_load(factoextra)
p_load(FactoMineR)
p_load(skimr)
p_load(stringr)
p_load(umap)
p_load(dbscan)
p_load(ggdensity)
p_load(forcats)
p_load(usemodels)
p_load(glmnet,kknn, earth, kernlab,rules, baguette)
p_load(performance)


#reading training file
raw_dataframe = fread('C:/Users/pedro_jw08iyg/OneDrive/Área de Trabalho/DSA/Machine Learning/5_Regressao - parte 2/3-Cap05-Python/dados/treino.csv')
raw_dataframe = janitor::clean_names(raw_dataframe)


#1) Data Overview ----
glimpse(raw_dataframe)
skim(raw_dataframe)

cleaned_dataframe = copy(raw_dataframe)
cleaned_dataframe[,id:=NULL]
cleaned_dataframe[,alley:=NULL]
cleaned_dataframe[,fireplace_qu:=NULL]
cleaned_dataframe[,pool_qc:=NULL]
cleaned_dataframe[,fence:=NULL]
cleaned_dataframe[,misc_feature:=NULL]
cleaned_dataframe[,lot_frontage:=NULL]

#2) Missing Values ----
naniar::vis_miss(cleaned_dataframe, cluster = T)
naniar::gg_miss_case(cleaned_dataframe)



predictive_variables = c('sale_price','ms_sub_class','ms_zoning','lot_area','lot_shape','land_contour',
                         'neighborhood','bldg_type','house_style','overall_qual','overall_cond',
                         'year_built','year_remod_add','exterior1st','exterior2nd','mas_vnr_type',
                         'mas_vnr_area','exter_qual','foundation','bsmt_qual','bsmt_exposure',
                         'bsmt_fin_type1','bsmt_fin_sf1','bsmt_unf_sf','total_bsmt_sf',
                         'heating_qc','central_air','x1st_flr_sf','x2nd_flr_sf','gr_liv_area',
                         'bsmt_full_bath','full_bath','half_bath','bedroom_abv_gr',
                         'kitchen_abv_gr','kitchen_qual','tot_rms_abv_grd','functional',
                         'fireplaces','garage_type','garage_yr_blt','garage_finish',
                         'garage_cars','garage_area','paved_drive','wood_deck_sf','open_porch_sf')

cleaned_dataframe = cleaned_dataframe[,..predictive_variables]



# 3) EDA and Feature Engineering ----


# 3.1) House Area Features ----


#1st vs 2nd floor
ggplot(cleaned_dataframe) +
  aes(x = x1st_flr_sf, y = x2nd_flr_sf, color = log(sale_price) ) +
  geom_point(size = 0.5) +  
  geom_density2d() +  
  geom_smooth(method = "lm", formula = "y~x", col = "red") +  
  theme_minimal() +  
  labs(title = "Surface of 1st and 2nd floor")
#insight: both variables apparently have a moderate correlation when 2nd floor is present (area different than 0)


#correlation 1st an 2nd floor (r = -0.2)
cor(cleaned_dataframe$x1st_flr_sf,cleaned_dataframe$x2nd_flr_sf)

#correlation when 2nd floor is different than 0 (r = +0.43)
cor(cleaned_dataframe[x2nd_flr_sf!=0,x1st_flr_sf],cleaned_dataframe[x2nd_flr_sf!=0,x2nd_flr_sf])


#1st vs 2nd floor only cases when 2nd floor is present
ggplot(cleaned_dataframe[x2nd_flr_sf!=0]) +
  aes(x = x1st_flr_sf, y = x2nd_flr_sf, color = log(sale_price) ) +
  geom_point(size = 1.5, alpha = 0.6) +  
  geom_density2d() +  
  geom_smooth(method = "lm", formula = "y~x", col = "red",se = FALSE, linetype = 'dashed') +  
  theme_minimal() +  
  labs(title = "Surface of 1st and 2nd floor")+
  scale_color_viridis_c(option = 'inferno')


# Surface of 2nd floor distribution
ggplot(cleaned_dataframe) +
  aes(x = x2nd_flr_sf) +
  geom_histogram(bins = 30L) +
  theme_minimal()
#insight: most of the places do not have 2nd floor 


#creating total hoouse area variable 
cleaned_dataframe[,mas_vnr_area:=ifelse(is.na(mas_vnr_area),0,mas_vnr_area)]
cleaned_dataframe[,house_total_area:=x1st_flr_sf + 
                    x2nd_flr_sf + 
                    gr_liv_area + 
                    total_bsmt_sf + 
                    garage_area + 
                    mas_vnr_area + 
                    wood_deck_sf + 
                    open_porch_sf]

#proportion by each part of the house
cleaned_dataframe[,prop_x1st_flr_sf:=x1st_flr_sf/house_total_area]
cleaned_dataframe[,prop_x2nd_flr_sf:=x2nd_flr_sf/house_total_area]
cleaned_dataframe[,prop_gr_liv_area:=gr_liv_area/house_total_area]
cleaned_dataframe[,prop_total_bsmt_sf:=total_bsmt_sf/house_total_area]
cleaned_dataframe[,prop_garage_area:=garage_area/house_total_area]
cleaned_dataframe[,prop_mas_vnr_area:=mas_vnr_area/house_total_area]
cleaned_dataframe[,prop_wood_deck_sf:=wood_deck_sf/house_total_area]
cleaned_dataframe[,prop_open_porch_sf:=open_porch_sf/house_total_area]

#excluding total areas
cleaned_dataframe[,x1st_flr_sf:=NULL]
cleaned_dataframe[,x2nd_flr_sf:=NULL]
cleaned_dataframe[,gr_liv_area:=NULL]
cleaned_dataframe[,garage_area:=NULL]
cleaned_dataframe[,mas_vnr_area:=NULL]
cleaned_dataframe[,wood_deck_sf:=NULL]
cleaned_dataframe[,open_porch_sf:=NULL]


#lot area vs sale_pice
g = ggplot(cleaned_dataframe) +
  aes(x = log(lot_area), y = log(sale_price) ) +
  geom_point(size = 1.5, alpha = 0.6) +  
  geom_density2d() +  
  geom_smooth(method = "lm", formula = "y~x", col = "red",se = FALSE, linetype = 'dashed') +  
  theme_minimal() +  
  labs(title = "Lot Area vs Price")+
  scale_color_viridis_c(option = 'inferno')
ggplotly(g)

#lot price vs sale price : log correlation = 0.39
cor(log(cleaned_dataframe$sale_price),log(cleaned_dataframe$lot_area) )
#conclusion: lot has less influence on price than total house area


g = ggplot(cleaned_dataframe) +
  aes(x = log(lot_area), y = log(house_total_area), color = log(sale_price) ) +
  geom_point(size = 1.5, alpha = 0.6) +  
  geom_density2d() +  
  geom_smooth(method = "lm", formula = "y~x", col = "red",se = FALSE, linetype = 'dashed') +  
  theme_minimal() +  
  labs(title = "Lot Area vs Total House area")+
  scale_color_viridis_c(option = 'inferno')
ggplotly(g)

#tatal area vs lot ares: cor =0.42
cor(log(cleaned_dataframe$house_total_area),log(cleaned_dataframe$lot_area), use = "complete.obs" )


#3.2) Basement Features ----

#comparing finishing variables
g = ggplot(cleaned_dataframe[total_bsmt_sf!=0]) +
  aes(x = bsmt_fin_sf1, y = bsmt_unf_sf, color = log(sale_price) ) +
  geom_point(size = 1.5, alpha = 0.6) +  
  geom_density2d() +  
  geom_smooth(method = "lm", formula = "y~x", col = "red",se = FALSE, linetype = 'dashed') +  
  theme_minimal() +  
  labs(title = "Finished Type 1 vs Unfinished Area")+
  scale_color_viridis_c(option = 'inferno')

ggplotly(g)
#insight: both variables have no directly relationship


#total area vs unfished area
g = ggplot(cleaned_dataframe[total_bsmt_sf!=0]) +
  aes(x = total_bsmt_sf, y = bsmt_unf_sf, color = log(sale_price) ) +
  geom_point(size = 1.5, alpha = 0.6) +  
  geom_density2d() +  
  geom_smooth(method = "lm", formula = "y~x", col = "red",se = FALSE, linetype = 'dashed') +  
  theme_minimal() +  
  labs(title = "Tota Area vs Unfinished Area")+
  scale_color_viridis_c(option = 'inferno')

ggplotly(g)

#calculating percentage of basement area constructed
cleaned_dataframe[,bsmt_pct_finished:=ifelse(total_bsmt_sf!=0,1-bsmt_unf_sf/total_bsmt_sf,0)]
cleaned_dataframe[,bsmt_unf_sf:=NULL] #excluding redundant feature

#total area vs finshed area type 1
g = ggplot(cleaned_dataframe[total_bsmt_sf!=0]) +
  aes(x = total_bsmt_sf, y = bsmt_fin_sf1, color = log(sale_price) ) +
  geom_point(size = 1.5, alpha = 0.6) +  
  geom_density2d() +  
  geom_smooth(method = "lm", formula = "y~x", col = "red",se = FALSE, linetype = 'dashed') +  
  theme_minimal() +  
  labs(title = "Tota Area vs Finished Area type 1")+
  scale_color_viridis_c(option = 'inferno')

ggplotly(g)

#calculating percentage of basement area constructed type 1
cleaned_dataframe[,bsmt_pct_finished_type1:=ifelse(total_bsmt_sf!=0,bsmt_fin_sf1/total_bsmt_sf,0)]
cleaned_dataframe[,bsmt_fin_sf1:=NULL] #excluding redundant feature
cleaned_dataframe[,total_bsmt_sf:=NULL] #excluding redundant feature


#comparing percentage basement completed
g = ggplot(cleaned_dataframe) +
  aes(x = bsmt_pct_finished,y = bsmt_pct_finished_type1, color = log(sale_price) ) +
  geom_point(size = 1.5, alpha = 0.6) +  
  geom_density2d() +  
  geom_smooth(method = "lm", formula = "y~x", col = "red",se = FALSE, linetype = 'dashed') +  
  theme_minimal() +  
  labs(title = "Percent constructed vs area type 1 constructed of basement")+
  scale_color_viridis_c(option = 'inferno')

ggplotly(g)
#insights:
#both variables have a high correlation 
#there are a some differences, but it seems to not have much impact on target variable

#removing the more specific variable
cleaned_dataframe[,bsmt_pct_finished_type1:=NULL]

#"bsmt_pct_finished" distribution
qplot(data = cleaned_dataframe,x = bsmt_pct_finished,geom = 'histogram')


#creating basement status:
#0 => pending
#0 > p > 1 => work in progress
# 1 = completed
cleaned_dataframe[,bsmt_status:=fcase(
  
  bsmt_pct_finished==0, 'pending',
  bsmt_pct_finished==1, 'completed',
  default = 'wip'
  
  
)]

#excluding redundant feature
cleaned_dataframe[,bsmt_pct_finished:=NULL]



#3.3) Year Features ----

g = ggplot(cleaned_dataframe) +
  aes(x = year_built, y = year_remod_add, color = log(sale_price) ) +
  geom_point(size = 1.5, alpha = 0.6) +  
  geom_density2d() +  
  geom_smooth(method = "lm", formula = "y~x", col = "red",se = FALSE, linetype = 'dashed') +  
  theme_minimal() +  
  labs(title = "Built vs Remodeled Year")+
  scale_color_viridis_c(option = 'inferno')

ggplotly(g)
#insights:
#some houses were not reformed yet (yera romod  = year built)
#maybe the registers of Year Remodel of 1950 are not reliable (horizontal line)


#creating remodeling status
cleaned_dataframe[,was_remodeled:=fcase(
  
  year_remod_add == 1950, 'not_clear', #anomaly status
  year_remod_add == year_built, 'no',  #not remodeled
  year_remod_add > year_built, 'yes',  #remodeled at least once
  default = 'error'
  
)]


#remodeling vs. garage construction
g = ggplot(cleaned_dataframe) +
  aes(x = garage_yr_blt, y = year_remod_add, color = log(sale_price) ) +
  geom_point(size = 1.5, alpha = 0.6) +  
  geom_density2d() +  
  geom_smooth(method = "lm", formula = "y~x", col = "red",se = FALSE, linetype = 'dashed') +  
  theme_minimal() +  
  labs(title = "Built vs Remodeled Year")+
  scale_color_viridis_c(option = 'inferno')

ggplotly(g)
#inisghts:
#most of garages were built in remodeled year
#there is no additional pattern regarding year of garage construction

#excluding redundant feature
cleaned_dataframe[,year_remod_add:=NULL]

#excluding garage built year
cleaned_dataframe[,garage_yr_blt:=NULL]



#3.4) Quality Metrics ----


#kitchen quality vs basement quality
ggplot(cleaned_dataframe) +
  aes(x = bsmt_qual, fill = kitchen_qual) +
  geom_bar(position = "fill") +
  scale_fill_hue(direction = 1) +
  theme_minimal()

#comparing both variables with sales price
cleaned_dataframe |> 
  drop_na() |>  
  group_by(bsmt_qual, kitchen_qual) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(bsmt_qual, kitchen_qual, fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")
#insights:
#when both basement and kitchen are excellent, the prices are the highest
#basement poor condition seems to leading the prices down


#3.5) Exterior Materials 1st and 2nd floor ----
cleaned_dataframe |> 
  drop_na() |>  
  group_by(exterior1st, exterior2nd) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(exterior1st, exterior2nd, fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")
#specific combinations seems to have more influence on price
#creating new feature - in training part, we noticed that exterior 2nd was not actually relevant, so we are going to remove this variable
#cleaned_dataframe[,exterior_covering:=paste(exterior1st, exterior2nd, sep = "_")]

#excluding etxerior 2nd
cleaned_dataframe[,exterior2nd:=NULL]

table(cleaned_dataframe$exterior1st)
qplot(data = cleaned_dataframe, x = exterior1st, y = log(sale_price), geom = 'boxplot') 

#there are a lot of low frequency categories, so we are going to keep just the most frequent ones
cleaned_dataframe[,exterior_remodeled:=fcase(
  exterior1st=='VinylSd','VinylSd',
  exterior1st=='HdBoard','HdBoard',
  exterior1st=='Wd Sdng','Wd Sdng',
  exterior1st=='MetalSd','MetalSd',
  exterior1st=='CemntBd','CemntBd',
  
  default = 'other'
  
)]

#checking results
table(cleaned_dataframe$exterior1st, cleaned_dataframe$exterior_remodeled)
qplot(data = cleaned_dataframe, x = exterior_remodeled, y = log(sale_price), geom = 'boxplot') 
cleaned_dataframe[,exterior1st:=NULL] #excluding redundant feature

#3.6) Half vs Full Bath ----
cleaned_dataframe |> 
  drop_na() |>  
  group_by(half_bath, full_bath) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(half_bath, full_bath, fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")

qplot(data = cleaned_dataframe, x = forcats::fct_reorder(factor(half_bath),log(sale_price),.fun = median), y = log(sale_price), geom = 'boxplot') 
table(cleaned_dataframe$half_bath)

#binarizing half bath
cleaned_dataframe[,has_half_bath:=ifelse(half_bath>0,T,F)]
#insight: number of half bathrooms has no ordered relationship with price


qplot(data = cleaned_dataframe, x = forcats::fct_reorder(factor(full_bath),log(sale_price),.fun = median), y = log(sale_price),color = has_half_bath, geom = 'boxplot') 
table(cleaned_dataframe$full_bath)
qplot(data = cleaned_dataframe[full_bath > 0], x = forcats::fct_reorder(factor(full_bath),log(sale_price),.fun = median), y = log(sale_price),color = has_half_bath, geom = 'boxplot') 
#full baths have positive correlation with prices
#there are just 9 houses with 0 full baths! (maybe we should filter those cases)

cleaned_dataframe[,half_bath:=NULL] #excluding redundant feature



#3.7) year built vs ms_sub_class ----
g = ggplot(cleaned_dataframe) +
  aes(x = year_built, y = factor(ms_sub_class), color = log(sale_price) ) +
  geom_jitter(size = 1.5, alpha = 0.6) +  
  # geom_density2d() +  
  #geom_smooth(method = "lm", formula = "y~x", col = "red",se = FALSE, linetype = 'dashed') +  
  theme_minimal() +  
  labs(title = "year built vs ms sub class")+
  scale_color_viridis_c(option = 'inferno')
ggplotly(g)


#3.8) External variables

#neighborhood vs exterior covering
cleaned_dataframe |> 
  drop_na() |>  
  group_by(neighborhood, exterior_remodeled) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(neighborhood, exterior_remodeled, fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")

#neighborhood vs external quality
cleaned_dataframe |> 
  drop_na() |>  
  group_by(neighborhood, exter_qual) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(neighborhood, exter_qual, fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")
#insight: as expected, higher external quality leads to higher prices 

#specific price distributions by neighborhood
qplot(data = cleaned_dataframe, x = forcats::fct_reorder(neighborhood,log(sale_price),.fun = median), y = log(sale_price), geom = 'boxplot') 
table(cleaned_dataframe$neighborhood)


qplot(data = cleaned_dataframe, x = forcats::fct_reorder(neighborhood,year_built,.fun = median), y = year_built, geom = 'boxplot') 
#there are newer and older regions

#important neighborhoods via feature selection (boruta)
important_neighborhoods = c('Crawfor','Edwards','NridgHt','CollgCr','BrDale','ClearCr','NAmes','NPkVill','OldTown','NoRidge','MeadowV','StoneBr','Somerst','Gilbert','Sawyer','NWAmes')
cleaned_dataframe[,importance_neighborhood_flag:=ifelse(neighborhood %in% important_neighborhoods,T,F) ]

qplot(data = cleaned_dataframe, x = forcats::fct_reorder(neighborhood,year_built,.fun = median), y = year_built, color = importance_neighborhood_flag,geom = 'boxplot') 
qplot(data = cleaned_dataframe, x = forcats::fct_reorder(neighborhood,year_built,.fun = mean), y = year_built, color = importance_neighborhood_flag,geom = 'boxplot') 
qplot(data = cleaned_dataframe, x = forcats::fct_reorder(neighborhood,log(sale_price),.fun = median), y = log(sale_price),color = importance_neighborhood_flag, geom = 'boxplot') 
qplot(data = cleaned_dataframe, x = forcats::fct_reorder(neighborhood,prop_open_porch_sf,.fun = median), y = prop_open_porch_sf,color = importance_neighborhood_flag, geom = 'boxplot') 

cleaned_dataframe[,importance_neighborhood_flag:=NULL]


#4) Recoding variables ----


cleaned_dataframe |>skim()


# 4.1) lot_shape ----

unique(cleaned_dataframe[,lot_shape])
table(cleaned_dataframe[,lot_shape])
qplot(data = cleaned_dataframe, x = lot_shape, y = log(sale_price), geom = 'boxplot') 
#regular terrains tend to have higher prices
#no significant distinctions among  strata of irregularities

cleaned_dataframe[,lot_shape:=fcase(
  
  lot_shape=='Reg','regular',
  lot_shape=='IR1','irregular',
  lot_shape=='IR2','irregular',
  lot_shape=='IR3','irregular',
  default = 'error'
  
)]                 


#4.2) land_contour ----
unique(cleaned_dataframe[,land_contour])
table(cleaned_dataframe[,land_contour])
qplot(data = cleaned_dataframe, x = factor(land_contour), y = log(sale_price), geom = 'boxplot') 

cleaned_dataframe[,land_contour:=as.numeric(fcase(
  
  land_contour=='Low','-1',
  land_contour=='Lvl','0',
  land_contour=='Bnk','1',
  land_contour=='HLS','2',
  default = 'error'
  
))] 

# 4.3) house_style ----

qplot(data = cleaned_dataframe, x = factor(house_style), y = log(sale_price), geom = 'boxplot') 
#insight: significant difference of finished vs. unfinished


qplot(data = cleaned_dataframe, x = factor(house_style), y = prop_x2nd_flr_sf, geom = 'boxplot') 
#inisght:
#house style is linked with 2nd floor proportion
#there are some anomalies houses in 1Story and SFoyer (proportion 2nd > 0)

#house style vs ms_sub_class
table(cleaned_dataframe$house_style, cleaned_dataframe$ms_sub_class)

cleaned_dataframe |> 
  drop_na() |>  
  group_by(house_style, ms_sub_class) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(house_style, factor(ms_sub_class), fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")

#extracting only the type of house of "house_style" variable (does not matter if it is unifinished)
cleaned_dataframe[,house_style_type_house:=fcase(
  
  house_style=='1Story','1_story',
  house_style=='1.5Fin','1_and_half_story',
  house_style=='1.5Unf','1_and_half_story',
  house_style=='2.5Fin','2_and_half_story',
  house_style=='2.5Unf','2_and_half_story',
  house_style=='2Story','2_story',
  house_style=='SFoyer','split_foyer',
  house_style=='SLvl','split_level',
  
  
  default = 'error'
  
)]

#extracting house type from the 'ms_sub_class_type_house' variable
cleaned_dataframe[,ms_sub_class_type_house:=fcase(
  
  ms_sub_class==20,'1_story',
  ms_sub_class==30,'1_story',
  ms_sub_class==40,'1_story',
  ms_sub_class==90,'1_story', #duplex
  ms_sub_class==120,'1_story',
  ms_sub_class==45,'1_and_half_story',
  ms_sub_class==50,'1_and_half_story',
  ms_sub_class==75,'2_and_half_story',
  ms_sub_class==60,'2_story',
  ms_sub_class==70,'2_story',
  ms_sub_class==160,'2_story',
  ms_sub_class==85,'split_foyer',
  ms_sub_class==80,'split_level',
  ms_sub_class==190,'all_types_family_conversion', #not significant in price and number of register
  ms_sub_class==180,'split_foyer_or_level', #not significant (few registers)
  
  
  default = 'error'
  
)]

#analysing new variables of type of house
a = table(cleaned_dataframe$house_style_type_house, cleaned_dataframe$ms_sub_class_type_house)

#exporting to clipboard
write.table(a, file = 'clipboard', sep ='\t', quote = F, row.names = F)

cleaned_dataframe |> 
  drop_na() |>  
  group_by(house_style_type_house, ms_sub_class_type_house) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(house_style_type_house, ms_sub_class_type_house, fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")


#extracting finished information from "house_style" variable
cleaned_dataframe[,house_style_finished_status:=fcase(
  
  house_style=='1Story','unknown',
  house_style=='1.5Fin','finished',
  house_style=='1.5Unf','unfinished',
  house_style=='2.5Fin','finished',
  house_style=='2.5Unf','unfinished',
  house_style=='2Story','unknown',
  house_style=='SFoyer','unknown',
  house_style=='SLvl','unknown',
  
  default = 'error'
  
)]



cleaned_dataframe[,ms_sub_class_finished_status:=fcase(
  
  ms_sub_class==20,'unknown',
  ms_sub_class==30,'unknown',
  ms_sub_class==40,'finished',
  ms_sub_class==90,'duplex', #duplex
  ms_sub_class==120,'unfinished',
  ms_sub_class==45,'unfinished',
  ms_sub_class==50,'finished',
  ms_sub_class==75,'unknown',
  ms_sub_class==60,'unknown',
  ms_sub_class==70,'unknown',
  ms_sub_class==160,'unknown',
  ms_sub_class==85,'unknown',
  ms_sub_class==80,'unknown',
  ms_sub_class==190,'unknown', #not significant in price and number of register
  ms_sub_class==180,'unknown', #not significant (few registers)
  
  
  default = 'error'
  
)]


table(cleaned_dataframe$house_style_finished_status, cleaned_dataframe$ms_sub_class_finished_status)

cleaned_dataframe |> 
  drop_na() |>  
  group_by(house_style_finished_status, ms_sub_class_finished_status) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(house_style_finished_status, ms_sub_class_finished_status, fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")
#inisghts:
#variables do not agree if the house is finished or not
#beeing duplex seems to be a high valuable factor, but it is rare

#unifying finishing status
cleaned_dataframe[,unified_finished_status:=fcase(
  house_style_finished_status == 'finished' |  ms_sub_class_finished_status == 'finished', 'finished',
  house_style_finished_status == 'unfinished' |  ms_sub_class_finished_status == 'unfinished', 'unfinished', #it can overwrite the finished status
  
  default = 'unknown'
  
)]

#checking new variable
table(cleaned_dataframe$unified_finished_status, cleaned_dataframe$ms_sub_class_finished_status)

table(cleaned_dataframe$unified_finished_status, cleaned_dataframe$house_style_finished_status)


#removing redundant variables
cleaned_dataframe[,ms_sub_class_type_house:=NULL] #almost the same as house_style_finished_status
cleaned_dataframe[,ms_sub_class_finished_status:=NULL]
cleaned_dataframe[,ms_sub_class:=NULL]
cleaned_dataframe[,house_style_finished_status:=NULL]
cleaned_dataframe[,house_style:=NULL]




#house style type vs bldg type ----
table(cleaned_dataframe$house_style_type_house, cleaned_dataframe$bldg_type)



cleaned_dataframe |> 
  drop_na() |>  
  group_by(house_style_type_house, bldg_type) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(house_style_type_house, bldg_type, fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")


# 4.4) external quality -----


cleaned_dataframe[,exter_qual_num:=as.numeric(fcase(
  exter_qual=='Po','-2',
  exter_qual=='Fa','-1',
  exter_qual=='TA','0',
  exter_qual=='Gd','1',
  exter_qual=='Ex','2',
  default = 'error'))]

table(cleaned_dataframe$exter_qual_num, cleaned_dataframe$exter_qual)
cleaned_dataframe[,exter_qual:=NULL]

# 4.5) basement condition -----
qplot(data = cleaned_dataframe, x = factor(bsmt_qual), y = log(sale_price), geom = 'boxplot') 

cleaned_dataframe[,bsmt_qual_num:=as.numeric(fcase(
  is.na(bsmt_qual),'0', #without basement
  bsmt_qual=='Po','1',
  bsmt_qual=='Fa','2',
  bsmt_qual=='TA','3',
  bsmt_qual=='Gd','4',
  bsmt_qual=='Ex','5',
  default = 'error'))]

table(cleaned_dataframe$bsmt_qual_num, cleaned_dataframe$bsmt_qual)
qplot(data = cleaned_dataframe, x = factor(bsmt_qual_num), y = log(sale_price), geom = 'boxplot') 
#insight: we can see there is a positive correlation between basement quality and price

cleaned_dataframe[,bsmt_qual:=NULL] #removing redundant variable

#4.6) BsmtFinType1: Rating of basement finished area

cleaned_dataframe[,bsmt_fin_type1_num:=as.numeric(fcase(
  is.na(bsmt_fin_type1),'0', #without basement
  bsmt_fin_type1=='Unf','1',
  bsmt_fin_type1=='LwQ','2',
  bsmt_fin_type1=='Rec','3',
  bsmt_fin_type1=='BLQ','4',
  bsmt_fin_type1=='ALQ','5',
  bsmt_fin_type1=='GLQ','6',
  default = 'error'))]

table(cleaned_dataframe$bsmt_fin_type1_num, cleaned_dataframe$bsmt_fin_type1)
qplot(data = cleaned_dataframe, x = factor(bsmt_fin_type1_num), y = log(sale_price), geom = 'boxplot') 
#it seems that the variable has no ordinal effect for all classes
#based on the description, we're not sure if the levels are regarding the same aspects
cleaned_dataframe[,bsmt_fin_type1_num:=NULL] #excluding ordinal



#we can see that all missing bsmtm_fin_type1 are unifished (compared with bstm_status)
cleaned_dataframe |> 
  # drop_na() |>  
  group_by(bsmt_fin_type1, bsmt_status ) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(bsmt_fin_type1, bsmt_status, fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")


ggplot(cleaned_dataframe) +
  aes(x  = forcats::fct_reorder(bsmt_fin_type1,log(sale_price),.fun = median), y = log(sale_price), color = bsmt_status)+
  geom_boxplot()
#inisghts:
#we can see that the missing values has significant lower prices than unfished basements. Maybe they are another category different than type 1 basement
#LwQ, BLQ, Rec have similar patterns.

#filling basement type 1 status
cleaned_dataframe[is.na(bsmt_fin_type1),bsmt_fin_type1:='not_type_1'] #filling missing registers

#recoding bsmt_fin_type1
cleaned_dataframe[,bsmt_fin_type1:=ifelse(
  bsmt_fin_type1 %in% c('ALQ','BLQ','LwQ','Rec'),'not_good', bsmt_fin_type1)
]

ggplot(cleaned_dataframe) +
  aes(x  = forcats::fct_reorder(bsmt_fin_type1,log(sale_price),.fun = median), y = log(sale_price), color = bsmt_status)+
  geom_boxplot()
#we can see a more clear pattern with this new engineered variable



# 4.6) HeatingQC: Heating quality and condition ----
cleaned_dataframe[,heating_qc_num:=as.numeric(fcase(
  heating_qc=='Po','-2',
  heating_qc=='Fa','-1',
  heating_qc=='TA','0',
  heating_qc=='Gd','1',
  heating_qc=='Ex','2',
  default = 'error'))]

table(cleaned_dataframe$heating_qc_num, cleaned_dataframe$heating_qc)
qplot(data = cleaned_dataframe, x = factor(heating_qc_num), y = log(sale_price), geom = 'boxplot') 
#we can notice a positive correlation

cleaned_dataframe[,heating_qc:=NULL] #excluding variable



#4.7) Kitchen quality ----
cleaned_dataframe[,kitchen_qual_num:=as.numeric(fcase(
  kitchen_qual=='Po','-2',
  kitchen_qual=='Fa','-1',
  kitchen_qual=='TA','0',
  kitchen_qual=='Gd','1',
  kitchen_qual=='Ex','2',
  default = 'error'))]

table(cleaned_dataframe$kitchen_qual_num, cleaned_dataframe$kitchen_qual)
qplot(data = cleaned_dataframe, x = factor(kitchen_qual_num), y = log(sale_price), geom = 'boxplot') 
#inisght: we can notice a high positive correlation

cleaned_dataframe[,kitchen_qual:=NULL] #excluding variable

#4.8) Functional: Home functionality (Assume typical unless deductions are warranted) ----
qplot(data = cleaned_dataframe, x = factor(functional), y = log(sale_price), geom = 'boxplot') 
#inisght: pattern not clear

#4.9) Garage Type ----
qplot(data = cleaned_dataframe, x = factor(garage_type), y = log(sale_price), geom = 'boxplot') 

#filling missing value
cleaned_dataframe[is.na(garage_type),garage_type:='no_garage']


#4.9) GarageFinish: Interior finish of the garage ----
qplot(data = cleaned_dataframe, x = factor(garage_finish), y = log(sale_price), geom = 'boxplot') 
cleaned_dataframe[,garage_finish_num:=as.numeric(fcase(
  is.na(garage_finish),'0', #without garage
  garage_finish=='Unf','1',
  garage_finish=='RFn','2',
  garage_finish=='Fin','3',
  default = 'error'))]

table(cleaned_dataframe$garage_finish_num, cleaned_dataframe$garage_finish)
qplot(data = cleaned_dataframe, x = factor(garage_finish_num), y = log(sale_price), geom = 'boxplot') 
cleaned_dataframe[,garage_finish:=NULL] #excluding original variable
#inisght: we can notice a positive correlation

#PavedDrive: Paved driveway

cleaned_dataframe[,paved_drive_num:=as.numeric(fcase(
  paved_drive=='N','0',
  paved_drive=='P','1',
  paved_drive=='Y','2',
  default = 'error'))]

table(cleaned_dataframe$paved_drive_num, cleaned_dataframe$paved_drive)
qplot(data = cleaned_dataframe, x = factor(paved_drive_num), y = log(sale_price), geom = 'boxplot') 
#inisght: slightly positive correlation

cleaned_dataframe[,paved_drive:=NULL]

# 4.10 bstm status -----

qplot(data = cleaned_dataframe, x = factor(bsmt_status), y = log(sale_price), geom = 'boxplot') 
#insight: it seems that it does not have significant impact on price


#4.11) Overall quality and overall condition ----

cleaned_dataframe |> 
  drop_na() |>  
  group_by(overall_cond, overall_qual) |> 
  summarise(sales_log_median = median(log(sale_price), na.rm = TRUE)) |> 
  ggplot(aes(factor(overall_cond), factor(overall_qual), fill = sales_log_median)) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_minimal() +
  theme(legend.position = "top")


cleaned_dataframe |> 
  drop_na() |>  
  ggplot(aes(x = factor(overall_cond), y = log(sale_price))) +
  geom_boxplot() 
#inisght: overall condition has a slightly postive correlation in the intervall 1 - 5

cleaned_dataframe |> 
  drop_na() |>  
  ggplot(aes(x = factor(overall_qual), y = log(sale_price))) +
  geom_boxplot() 
#insight: overall condition has a high positive correlation wit price

#comapring both scores via linear model
lm(data = cleaned_dataframe, formula= log(sale_price) ~ overall_qual + overall_cond) |> summary()
#overall_qual has a strong importance than overall quality


cleaned_dataframe[,overall_unified:=0.013850*overall_cond+0.237052*overall_qual+10]
cleaned_dataframe |> 
  drop_na() |>  
  ggplot(aes(x = factor(overall_unified), y = log(sale_price))) +
  geom_boxplot() 

cleaned_dataframe[,overall_unified:=NULL] #excluding new unified factor
#inisght: we are going to keep both quality scores as separated variables


#4.12) Functional ----


table(cleaned_dataframe$functional)
#this variable is not well balanced

#extracting information about if there was damage or not
cleaned_dataframe[,functional_typical:=ifelse(functional=='Typ',T,F)]
cleaned_dataframe[,functional:=NULL] #excluding redundant variable
qplot(data = cleaned_dataframe, x = functional_typical, y = log(sale_price), geom = 'boxplot') 


# 4.13) Foundation ----
table(cleaned_dataframe$foundation)
#very imbalanced category

qplot(data = cleaned_dataframe, x = forcats::fct_reorder(factor(foundation),log(sale_price),.fun = median), y = log(sale_price), geom = 'boxplot') 
cleaned_dataframe[,foundation:=fcase(
  foundation=='BrkTil','BrkTil',
  foundation=='CBlock','CBlock',
  foundation=='PConc','PConc',
  default = 'other' #low registers categories
  
)]



#5) Dealing with missing values ----

#overview remaining missing value
naniar::vis_miss(cleaned_dataframe, cluster = T)
naniar::gg_miss_case(cleaned_dataframe)


#5.1) mas_vnr_type ----
raw_dataframe[is.na(mas_vnr_type),mas_vnr_area]
#all missing values are when there is no area, so we will replace manually missing values

cleaned_dataframe[is.na(mas_vnr_type),mas_vnr_type:='none']

#binning target for equal split train/test
# cleaned_dataframe[,binned_target:=ntile(sale_price,n = 5)]
cleaned_dataframe[,sale_price_temp:=log(sale_price,base = 10)]

# cleaned_dataframe[,has_half_bath:=has_half_bath*1]
# cleaned_dataframe[,functional_typical:=functional_typical*1]





#6) Baseline Model ----

#removing unimportant features
cleaned_dataframe[,land_contour:=NULL]
cleaned_dataframe[,was_remodeled:=NULL]
cleaned_dataframe[,bsmt_exposure:=NULL]

# 6.1) DATA RESAMPLING ----

set.seed(123)
cleaned_dataframe_split = rsample::initial_split(data = cleaned_dataframe,
                                                 prop = 0.80,strata = sale_price_temp )

cleaned_dataframe_training = cleaned_dataframe_split |> training()
cleaned_dataframe_validation = cleaned_dataframe_split |> testing()

#6.2) MODEL SPECIFICATION ----

baseline_model = linear_reg() |>
  # Set the engine
  set_engine("lm")


# baseline_model = 
#   boost_tree() |>
#   set_mode("regression") |>
#   set_engine("xgboost")

# baseline_model = 
#   mlp() |> 
#   set_engine("nnet", trace = 0) |> 
#   set_mode("regression")


# baseline_model = parsnip::rand_forest() |>
#   # Set the engine
#   set_engine("ranger") |>
#   set_mode('regression')


#6.3) FEATURE ENGINEERING ----


recipe_spec = recipes::recipe(sale_price~.,
                              data = cleaned_dataframe_training) |>
  step_range(paved_drive_num,garage_finish_num,overall_qual,kitchen_qual_num,
             overall_cond,year_built,exter_qual_num,full_bath,heating_qc_num,
             bsmt_qual_num,overall_qual,tot_rms_abv_grd,
             bsmt_full_bath,bedroom_abv_gr,kitchen_abv_gr,tot_rms_abv_grd,
             fireplaces,garage_cars) |> #categorical ordinal encoded as numbers
  step_pca(starts_with('prop_'),num_comp = 1) |> #compress prop variables
  step_log(lot_area,house_total_area,sale_price, base = 10) |> #do not put the target variable explicitly here
  step_normalize(lot_area,house_total_area) |> 
  step_integer(has_half_bath,functional_typical) |> 
  # step_dummy(all_nominal(),-all_outcomes()) |> 
  step_rm(neighborhood) |> #removed neighborhood because we want more general model
  step_dummy(all_nominal_predictors()) |> 
  step_rm(sale_price_temp,ms_zoning_RH,bldg_type_X2fmCon,mas_vnr_type_none,foundation_other,garage_type_Basment,
          exterior_remodeled_other,house_style_type_house_X2_and_half_story,house_style_type_house_split_foyer) #removing variables with low predictive power

#glimpse transformations
recipe_spec |> prep() |> juice() |> glimpse()



#6.4) RECIPE TRAINING ----

recipe_prep_baseline = recipe_spec |> 
  prep(training = cleaned_dataframe_training)

#6.5) PREPROCESS TRAINING DATA ----

cleaned_dataframe_training_prep = recipe_prep_baseline |>
  recipes::bake(new_data = NULL)


#6.7) PREPROCESS TEST DATA ----

cleaned_dataframe_validation_prep = recipe_prep_baseline |> 
  recipes::bake(new_data = cleaned_dataframe_validation)


#6.8) MODELS FITTING ----

baseline_model_fit = baseline_model |>
  parsnip::fit(sale_price ~ .,
               data = cleaned_dataframe_training_prep)




#6.9) PREDICTIONS ON VALIDATION DATA ----

predictions_baseline = predict(baseline_model_fit,
                               new_data = cleaned_dataframe_validation_prep)


setDT(predictions_baseline)

predictions_baseline = data.table(.pred = predictions_baseline$.pred,true_value = cleaned_dataframe_validation_prep$sale_price)
qplot(data = predictions_baseline,x = true_value, y =.pred,geom = 'point')

predictions_baseline |>  yardstick::rmse(truth = true_value, estimate = .pred)
predictions_baseline |>  yardstick::rsq(truth = true_value, estimate = .pred)

#the baseline model seems good, now we're going to try other models

#7) Screening other models via cross-validation ----


#7.1) Model Specification ----

#Elastic Net
model_spec_elastic_net =
  linear_reg(penalty = tune(), mixture = tune())  |>  
  set_engine("glmnet")


#Single layer neural network
model_spec_nnet = 
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) |> 
  set_engine("nnet", MaxNWts = 2600) |> 
  set_mode("regression")

#Ensembles of MARS models	
model_spec_mars = 
  mars(prod_degree = tune()) |>  #= use GCV to choose terms
  set_engine("earth") |> 
  set_mode("regression")

#Radial basis function support vector machines	
model_spec_svm_rbf = 
  svm_rbf(cost = tune(), rbf_sigma = tune()) |> 
  set_engine("kernlab") |> 
  set_mode("regression")

#Polynomial support vector machines	
model_spec_svm_poly = 
  svm_poly(cost = tune(), degree = tune()) |> 
  set_engine("kernlab") |> 
  set_mode("regression")

#K-nearest neighbors	
model_spec_knn = 
  nearest_neighbor(neighbors = tune(), dist_power = tune(), weight_func = tune()) |> 
  set_engine("kknn") |> 
  set_mode("regression")

#Decision trees	
model_spec_cart = 
  decision_tree(cost_complexity = tune(), min_n = tune()) |> 
  set_engine("rpart") |> 
  set_mode("regression")

#Ensembles of decision trees	
model_spec_ensemble_tree = 
  bag_tree() |> 
  set_engine("rpart", times = 50L) |> 
  set_mode("regression")

#Random forest	
model_spec_rf = 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |> 
  set_engine("ranger") |> 
  set_mode("regression")

#xgboost
model_spec_xgboost = 
  boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(), 
             min_n = tune(), sample_size = tune(), trees = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

#cubist
model_spec_cubist = 
  cubist_rules(committees = tune(), neighbors = tune()) |> 
  set_engine("Cubist")



#7.2) Workflow set ----

workflow_set_models_evaluation = 
  workflow_set(
    preproc = list(basic_pre_processor = recipe_spec), 
    models = list(
      # cart = model_spec_cart,
      # cubist = model_spec_cubist,
      # elastic_net = model_spec_elastic_net,
      # ensemble_tree = model_spec_ensemble_tree,
      # knn = model_spec_knn,
      # mars = model_spec_mars,
      # neural_network = model_spec_nnet,
      # random_forest = model_spec_rf,
      # svm_polynomial = model_spec_svm_poly,
      # svm_radial = model_spec_svm_rbf
      #xgboost = model_spec_xgboost
    )
  )




#we are going to use the whole cleaned dataset (originally the train dataset)
#hyperameters will be also tuned
set.seed(123)
folds_spec =  vfold_cv(cleaned_dataframe,
                       strata = sale_price, 
                       repeats = 5)


grid_ctrl = 
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )

p_load(tictoc)

tic()
grid_results = 
  workflow_set_models_evaluation |> 
  workflow_map(
    seed = 123,
    resamples = folds_spec,
    grid = 20,
    control = grid_ctrl
  )
toc()



grid_results %>% 
  rank_results() %>% 
  filter(.metric == "rmse") %>% 
  select(model, .config, rmse = mean, rank)


#saving the results of cross validation
#write_rds(grid_results,"C:/Users/pedro_jw08iyg/OneDrive/Área de Trabalho/DSA/Machine Learning/5_Regressao - parte 2/3-Cap05-Python/dados/validacao_svmrad_svmpoly.rds")

#loading all workflows
workflow_results_pt1 = read_rds("C:/Users/pedro_jw08iyg/OneDrive/Área de Trabalho/DSA/Machine Learning/5_Regressao - parte 2/3-Cap05-Python/dados/validacao_cart_cubist_elasticnet_knn_mars_rf.rds")
workflow_results_pt2 = read_rds("C:/Users/pedro_jw08iyg/OneDrive/Área de Trabalho/DSA/Machine Learning/5_Regressao - parte 2/3-Cap05-Python/dados/validacao_xgboost.rds")
workflow_results_pt3 = read_rds("C:/Users/pedro_jw08iyg/OneDrive/Área de Trabalho/DSA/Machine Learning/5_Regressao - parte 2/3-Cap05-Python/dados/validacao_svmrad_svmpoly.rds")



workflow_complete_fit = bind_rows(workflow_results_pt1,workflow_results_pt2,workflow_results_pt3)


#best workflows
workflow_complete_fit %>% 
  rank_results() %>% 
  filter(.metric == "rmse") %>% 
  select(model, .config, rmse = mean, rank)


#7.3) plotting the workflows root mean squared errors ----
autoplot(
  workflow_complete_fit,
  rank_metric = "rmse",  
  metric = "rmse",       
  select_best = TRUE    
) |> ggplotly()

#7.3) plotting the workflows r-squared ----

autoplot(
  workflow_complete_fit,
  rank_metric = "rsq",  
  metric = "rsq",       
  select_best = TRUE    
) |> ggplotly()


#7.4) Selecting the Best Workflow -----

best_wkflow_id = workflow_complete_fit |>  
  rank_results(
    rank_metric = 'rmse',
    select_best = TRUE
  ) |> 
  slice(1) |> 
  pull(wflow_id)

wkflow_best = extract_workflow(workflow_complete_fit, id = best_wkflow_id)

#7.5) #Extracting tuned results from the best Workflowset ----
wkflow_best_tuned = workflow_complete_fit[workflow_complete_fit$wflow_id == best_wkflow_id,
                                          "result"][[1]][[1]]



collect_metrics(wkflow_best_tuned)
autoplot(wkflow_best_tuned)

#best tuned parameters
select_best(wkflow_best_tuned, "rmse")


# 7.6) Finalizing Model With the best Hyperparameters ----
wkflow_best_final = finalize_workflow(x = wkflow_best, parameters = select_best(wkflow_best_tuned, "rmse"))


# 7.7) Finalizing Workflow -----

#Training Workflow with Training Data

wkflow_best_final_fitted = wkflow_best_final |>  fit(cleaned_dataframe)


#7.8) Loading test data and applying manual preprocessing ----
setwd('C:\\Users\\pedro_jw08iyg\\OneDrive\\Área de Trabalho\\DSA\\Projetos\\Machine Learning\\house_price')

#reading testing file
cleaned_test = fread('C:/Users/pedro_jw08iyg/OneDrive/Área de Trabalho/DSA/Machine Learning/5_Regressao - parte 2/3-Cap05-Python/dados/teste.csv')
cleaned_test = janitor::clean_names(cleaned_test)

cleaned_test = copy(raw_dataframe)
cleaned_test[,id:=NULL]
cleaned_test[,alley:=NULL]
cleaned_test[,fireplace_qu:=NULL]
cleaned_test[,pool_qc:=NULL]
cleaned_test[,fence:=NULL]
cleaned_test[,misc_feature:=NULL]
cleaned_test[,lot_frontage:=NULL]

cleaned_test = cleaned_test[,..predictive_variables]
cleaned_test[,mas_vnr_area:=ifelse(is.na(mas_vnr_area),0,mas_vnr_area)]
cleaned_test[,house_total_area:=x1st_flr_sf + 
               x2nd_flr_sf + 
               gr_liv_area + 
               total_bsmt_sf + 
               garage_area + 
               mas_vnr_area + 
               wood_deck_sf + 
               open_porch_sf]

cleaned_test[,prop_x1st_flr_sf:=x1st_flr_sf/house_total_area]
cleaned_test[,prop_x2nd_flr_sf:=x2nd_flr_sf/house_total_area]
cleaned_test[,prop_gr_liv_area:=gr_liv_area/house_total_area]
cleaned_test[,prop_total_bsmt_sf:=total_bsmt_sf/house_total_area]
cleaned_test[,prop_garage_area:=garage_area/house_total_area]
cleaned_test[,prop_mas_vnr_area:=mas_vnr_area/house_total_area]
cleaned_test[,prop_wood_deck_sf:=wood_deck_sf/house_total_area]
cleaned_test[,prop_open_porch_sf:=open_porch_sf/house_total_area]

cleaned_test[,x1st_flr_sf:=NULL]
cleaned_test[,x2nd_flr_sf:=NULL]
cleaned_test[,gr_liv_area:=NULL]
cleaned_test[,garage_area:=NULL]
cleaned_test[,mas_vnr_area:=NULL]
cleaned_test[,wood_deck_sf:=NULL]
cleaned_test[,open_porch_sf:=NULL]

cleaned_test[,bsmt_pct_finished:=ifelse(total_bsmt_sf!=0,1-bsmt_unf_sf/total_bsmt_sf,0)]
cleaned_test[,bsmt_unf_sf:=NULL] 
cleaned_test[,bsmt_pct_finished_type1:=ifelse(total_bsmt_sf!=0,bsmt_fin_sf1/total_bsmt_sf,0)]
cleaned_test[,bsmt_fin_sf1:=NULL] 
cleaned_test[,total_bsmt_sf:=NULL] 
cleaned_test[,bsmt_pct_finished_type1:=NULL]

cleaned_test[,bsmt_status:=fcase(
  
  bsmt_pct_finished==0, 'pending',
  bsmt_pct_finished==1, 'completed',
  default = 'wip'
  
  
)]

cleaned_test[,bsmt_pct_finished:=NULL]

cleaned_test[,was_remodeled:=fcase(
  
  year_remod_add == 1950, 'not_clear', #anomaly status
  year_remod_add == year_built, 'no',  #not remodeled
  year_remod_add > year_built, 'yes',  #remodeled at least once
  default = 'error'
  
)]

cleaned_test[,year_remod_add:=NULL]
cleaned_test[,garage_yr_blt:=NULL]
cleaned_test[,exterior2nd:=NULL]
cleaned_test[,exterior_remodeled:=fcase(
  exterior1st=='VinylSd','VinylSd',
  exterior1st=='HdBoard','HdBoard',
  exterior1st=='Wd Sdng','Wd Sdng',
  exterior1st=='MetalSd','MetalSd',
  exterior1st=='CemntBd','CemntBd',
  
  default = 'other'
  
)]

cleaned_test[,exterior1st:=NULL] #excluding redundant feature
cleaned_test[,has_half_bath:=ifelse(half_bath>0,T,F)]
cleaned_test[,half_bath:=NULL] #excluding redundant feature
cleaned_test[,importance_neighborhood_flag:=NULL]

cleaned_test[,lot_shape:=fcase(
  
  lot_shape=='Reg','regular',
  lot_shape=='IR1','irregular',
  lot_shape=='IR2','irregular',
  lot_shape=='IR3','irregular',
  default = 'error'
  
)]                 

cleaned_test[,land_contour:=as.numeric(fcase(
  
  land_contour=='Low','-1',
  land_contour=='Lvl','0',
  land_contour=='Bnk','1',
  land_contour=='HLS','2',
  default = 'error'
  
))] 

cleaned_test[,house_style_type_house:=fcase(
  
  house_style=='1Story','1_story',
  house_style=='1.5Fin','1_and_half_story',
  house_style=='1.5Unf','1_and_half_story',
  house_style=='2.5Fin','2_and_half_story',
  house_style=='2.5Unf','2_and_half_story',
  house_style=='2Story','2_story',
  house_style=='SFoyer','split_foyer',
  house_style=='SLvl','split_level',
  
  
  default = 'error'
  
)]

cleaned_test[,ms_sub_class_type_house:=fcase(
  
  ms_sub_class==20,'1_story',
  ms_sub_class==30,'1_story',
  ms_sub_class==40,'1_story',
  ms_sub_class==90,'1_story', #duplex
  ms_sub_class==120,'1_story',
  ms_sub_class==45,'1_and_half_story',
  ms_sub_class==50,'1_and_half_story',
  ms_sub_class==75,'2_and_half_story',
  ms_sub_class==60,'2_story',
  ms_sub_class==70,'2_story',
  ms_sub_class==160,'2_story',
  ms_sub_class==85,'split_foyer',
  ms_sub_class==80,'split_level',
  ms_sub_class==190,'all_types_family_conversion', #not significant in price and number of register
  ms_sub_class==180,'split_foyer_or_level', #not significant (few registers)
  
  
  default = 'error'
  
)]


cleaned_test[,house_style_finished_status:=fcase(
  
  house_style=='1Story','unknown',
  house_style=='1.5Fin','finished',
  house_style=='1.5Unf','unfinished',
  house_style=='2.5Fin','finished',
  house_style=='2.5Unf','unfinished',
  house_style=='2Story','unknown',
  house_style=='SFoyer','unknown',
  house_style=='SLvl','unknown',
  
  default = 'error'
  
)]



cleaned_test[,ms_sub_class_finished_status:=fcase(
  
  ms_sub_class==20,'unknown',
  ms_sub_class==30,'unknown',
  ms_sub_class==40,'finished',
  ms_sub_class==90,'duplex', #duplex
  ms_sub_class==120,'unfinished',
  ms_sub_class==45,'unfinished',
  ms_sub_class==50,'finished',
  ms_sub_class==75,'unknown',
  ms_sub_class==60,'unknown',
  ms_sub_class==70,'unknown',
  ms_sub_class==160,'unknown',
  ms_sub_class==85,'unknown',
  ms_sub_class==80,'unknown',
  ms_sub_class==190,'unknown', #not significant in price and number of register
  ms_sub_class==180,'unknown', #not significant (few registers)
  
  
  default = 'error'
  
)]

cleaned_test[,unified_finished_status:=fcase(
  house_style_finished_status == 'finished' |  ms_sub_class_finished_status == 'finished', 'finished',
  house_style_finished_status == 'unfinished' |  ms_sub_class_finished_status == 'unfinished', 'unfinished', #it can overwrite the finished status
  
  default = 'unknown'
  
)]


cleaned_test[,ms_sub_class_type_house:=NULL] #almost the same as house_style_finished_status
cleaned_test[,ms_sub_class_finished_status:=NULL]
cleaned_test[,ms_sub_class:=NULL]
cleaned_test[,house_style_finished_status:=NULL]
cleaned_test[,house_style:=NULL]

cleaned_test[,exter_qual_num:=as.numeric(fcase(
  exter_qual=='Po','-2',
  exter_qual=='Fa','-1',
  exter_qual=='TA','0',
  exter_qual=='Gd','1',
  exter_qual=='Ex','2',
  default = 'error'))]

cleaned_test[,exter_qual:=NULL]


cleaned_test[,bsmt_qual_num:=as.numeric(fcase(
  is.na(bsmt_qual),'0', #without basement
  bsmt_qual=='Po','1',
  bsmt_qual=='Fa','2',
  bsmt_qual=='TA','3',
  bsmt_qual=='Gd','4',
  bsmt_qual=='Ex','5',
  default = 'error'))]

cleaned_test[,bsmt_qual:=NULL] #removing redundant variable

cleaned_test[,bsmt_fin_type1_num:=as.numeric(fcase(
  is.na(bsmt_fin_type1),'0', #without basement
  bsmt_fin_type1=='Unf','1',
  bsmt_fin_type1=='LwQ','2',
  bsmt_fin_type1=='Rec','3',
  bsmt_fin_type1=='BLQ','4',
  bsmt_fin_type1=='ALQ','5',
  bsmt_fin_type1=='GLQ','6',
  default = 'error'))]

cleaned_test[,bsmt_fin_type1_num:=NULL] #excluding ordinal
cleaned_test[is.na(bsmt_fin_type1),bsmt_fin_type1:='not_type_1'] #filling missing registers
cleaned_test[,bsmt_fin_type1:=ifelse(
  bsmt_fin_type1 %in% c('ALQ','BLQ','LwQ','Rec'),'not_good', bsmt_fin_type1)
]

cleaned_test[,heating_qc_num:=as.numeric(fcase(
  heating_qc=='Po','-2',
  heating_qc=='Fa','-1',
  heating_qc=='TA','0',
  heating_qc=='Gd','1',
  heating_qc=='Ex','2',
  default = 'error'))]

cleaned_test[,heating_qc:=NULL] #excluding variable

cleaned_test[,kitchen_qual_num:=as.numeric(fcase(
  kitchen_qual=='Po','-2',
  kitchen_qual=='Fa','-1',
  kitchen_qual=='TA','0',
  kitchen_qual=='Gd','1',
  kitchen_qual=='Ex','2',
  default = 'error'))]

cleaned_test[,kitchen_qual:=NULL] #excluding variable
cleaned_test[is.na(garage_type),garage_type:='no_garage']
cleaned_test[,garage_finish_num:=as.numeric(fcase(
  is.na(garage_finish),'0', #without garage
  garage_finish=='Unf','1',
  garage_finish=='RFn','2',
  garage_finish=='Fin','3',
  default = 'error'))]
cleaned_test[,garage_finish:=NULL] #excluding original variable

cleaned_test[,paved_drive_num:=as.numeric(fcase(
  paved_drive=='N','0',
  paved_drive=='P','1',
  paved_drive=='Y','2',
  default = 'error'))]

cleaned_test[,paved_drive:=NULL]

cleaned_test[,functional_typical:=ifelse(functional=='Typ',T,F)]
cleaned_test[,functional:=NULL] #excluding redundant variable

cleaned_test[,foundation:=fcase(
  foundation=='BrkTil','BrkTil',
  foundation=='CBlock','CBlock',
  foundation=='PConc','PConc',
  default = 'other' #low registers categories
  
)]

cleaned_test[is.na(mas_vnr_type),mas_vnr_type:='none']
cleaned_test[,sale_price_temp:=log(sale_price,base = 10)]
cleaned_test[,land_contour:=NULL]
cleaned_test[,was_remodeled:=NULL]
cleaned_test[,bsmt_exposure:=NULL]

# 7.9) Evaluating performance on Test Data (error) ----

#comparing testing and training columns

names(cleaned_dataframe) %in% names(cleaned_test)
names(cleaned_test)[!names(cleaned_test) %in% names(cleaned_dataframe)]

wkflow_best_final_fitted |> predict(cleaned_dataframe)
#error explanation: unfortunately, recipe is not recognizing that sale_price is a outcome in step log
#we need to redo the recipe

#7.10) Adjusting step log for target variable in recipe -----

#adjusting recipe for predicting with target variable
new_recipe_spec = recipes::recipe(sale_price~.,
                                  data = cleaned_dataframe_training) |>
  step_range(paved_drive_num,garage_finish_num,overall_qual,kitchen_qual_num,
             overall_cond,year_built,exter_qual_num,full_bath,heating_qc_num,
             bsmt_qual_num,overall_qual,tot_rms_abv_grd,
             bsmt_full_bath,bedroom_abv_gr,kitchen_abv_gr,tot_rms_abv_grd,
             fireplaces,garage_cars) |> #categorical ordinal encoded as numbers
  step_pca(starts_with('prop_'),num_comp = 1) |> #compress prop variables
  step_log(lot_area,house_total_area, base = 10) |>
  step_log(all_outcomes(), base = 10, skip = TRUE) |> #target variable step. Set argument skip = T
  step_normalize(lot_area,house_total_area) |> 
  step_integer(has_half_bath,functional_typical) |> 
  step_rm(neighborhood) |> #removed neighborhood because we want more general model
  step_dummy(all_nominal_predictors()) |> 
  step_rm(sale_price_temp,ms_zoning_RH,bldg_type_X2fmCon,mas_vnr_type_none,foundation_other,garage_type_Basment,
          exterior_remodeled_other,house_style_type_house_X2_and_half_story,house_style_type_house_split_foyer) #removing variables with low predictive power


#creating the model specification with best hyperparameters
select_best(wkflow_best_tuned, "rmse")

new_model_spec_xgboost = 
  boost_tree(tree_depth = 12, 
             learn_rate = 0.00615,
             loss_reduction = 0.00000264, 
             min_n = 15,
             sample_size = 0.390, 
             trees = 1920) |> 
  set_engine("xgboost") |> 
  set_mode("regression")


#creating workflow
wkflow_best_final_updated = 
  workflow() |> 
  add_model(new_model_spec_xgboost) |> 
  add_recipe(new_recipe_spec)

#fitting best workflow with trainign data
wkflow_best_final_updated_fitted = wkflow_best_final_updated |> fit(cleaned_dataframe)

#predicting on test data
test_predictions = wkflow_best_final_updated_fitted |> predict(cleaned_test)

#inserting prediction on the dataframe
cleaned_test = cbind(cleaned_test,test_predictions)

#rmse and r-squared for transformed data
cleaned_test |> yardstick::rmse(truth = sale_price_temp, estimate = .pred)
cleaned_test |> yardstick::rsq(truth = sale_price_temp, estimate = .pred)


#calculating metrics for original scale 
cleaned_test[,pred_original_scale:=10^.pred]
cleaned_test |> yardstick::rmse(truth = sale_price, estimate = pred_original_scale)
cleaned_test |> yardstick::rsq(truth = sale_price, estimate = pred_original_scale)

ggplot(cleaned_test) +
  aes(x = pred_original_scale, y = sale_price) +
  geom_point()

ggplot(cleaned_test) +
  aes(y = sale_price) +
  geom_boxplot()

#7.11) Variable Importance -----

p_load(vip)
extract_fit_parsnip(wkflow_best_final_updated_fitted)  |> 
  vip(geom = "col")

#7.12) Checking residuals


tem_dataframe = copy(cleaned_test)

tem_dataframe[,residuos:=log(sale_price,base = 10)-.pred]  

(ggplot(tem_dataframe) + 
    aes(x = residuos) + 
    geom_histogram() )|>  ggplotly()


(ggplot(tem_dataframe) + 
    aes(y = residuos) + 
    geom_boxplot() )|>  ggplotly()

tem_dataframe[,low_pred:=ifelse(residuos< -0.06,T,F)]

tem_dataframe[,sale_price:=NULL]
tem_dataframe[,pred_original_scale:=NULL]
tem_dataframe[,.pred:=NULL]
tem_dataframe[,sale_price_temp:=NULL]
tem_dataframe[,residuos:=NULL]

