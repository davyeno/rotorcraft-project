import streamlit as st
from streamlit import caching
import pandas as pd
import numpy as np
import prophet as Prophet

import matplotlib as mpl

import altair as alt
import plotly as plt
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import base64
import itertools
from datetime import datetime

st.set_page_config(page_title ="Predict Hover App",
                    initial_sidebar_state="expanded",
                    layout="wide",
                    page_icon="chart_with_upwards_trend") #this can be replaced with any emoji

tabs = ['Physic Based Model']
        # "Forecast"] # define tabs

# Define column width control: This is the prevent the streamlit columns from colapsing too much and overlapping with each other.

def col_control():
    st.markdown(
        """
        <style>
        [data-testid="stHorizontalBlock"] > div:first-child {
            min-width: 300px;
        }
        [data-testid="stHorizontalBlock"] > div:nth-child(3) {
            min-width: 400px;
        }
        [data-testid="stHorizontalBlock"] > div:nth-child(5) {
            min-width: 400px;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )

# Set caching

@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner= True)

def load_csv(): # the function that formats the data on load
    df_input = pd.DataFrame()
    df_input = pd.read_csv(input, sep=None, engine='python', encoding='utf-8',
                           parse_dates=True,
                           infer_datetime_format=True)
    return df_input


def prep_data(df): # this function controls the column rename functionality and sorts the dataset
    df_input = df.rename({date_col: "ds", metric_col: "y"}, errors='raise', axis=1)
    st.markdown("The selected date column is now labeled as **ds** and the values column as **y**")
    df_input = df_input[['ds', 'y']]
    df_input = df_input.sort_values(by='ds', ascending=True)
    return df_input

def model_init(df,
               growth,
               n_changepoints,
               changepoint_scale,
               yearly_seasonality_scale,
               weekly_seasonality_scale,
               daily_seasonality_scale,
               seasonality,
               intervalw):
    m = Prophet.Prophet(growth=growth,
                        n_changepoints = n_changepoints,
                        seasonality_mode=seasonality,
                        changepoint_prior_scale=changepoint_scale,
                        yearly_seasonality = yearly_seasonality_scale,
                        weekly_seasonality = weekly_seasonality_scale,
                        daily_seasonality = daily_seasonality_scale,
                        interval_width=intervalw)
    m.fit(df)
    return m

# Goodness of fit functions
def mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred)**2)

def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))*100

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) / ((y_true + y_pred)/2))*100


# Clear cache on load
# caching.clear_cache()


# Define Dataframe and Img: This initiates the blank data from for upload, as well as prepared the dgi claims logo for display within the application
df =  pd.DataFrame()

#############
## Sidebar ##
#############

# Sidebar size increase: This st.markdown passes some CSS for adjusting the size of the sidebar. Increase or decreast the width and margin-left
# properties to adjust the size.

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar: # with st.sidebar calls the sidebar, and anything called 1 tab in below it will be within the sidebar.

    st.markdown("<h1 style='text-align: left; color: green;'>FAA Hover Classification Tool</h1>", unsafe_allow_html=True)
    # st.markdown("#")
    page = st.radio("Tabs",tabs)
    
    # input file
    input = st.file_uploader('')
    if input is None:
        st.write('Please input your file')
    if input: 
        with st.spinner('Loading data..'):
                df = load_csv()     

if page == 'Physic Based Model':
    
    main1, gap2, main2 =st.columns([1.2,.05,3])

    col_control() # this calls the declared formula for controlling the column widths
    if input is None:
            st.write("")
    
    if input:
        with main1:
            step2_text = '<p style="font-family:Arial; color:#7500C0; font-weight: bold; font-size: 25px; line-height: 15pt; font-variant: small-caps;">Model Parameters Tuning</p>'
            st.markdown(step2_text, unsafe_allow_html=True)
            st.markdown('<p style="font-family:Arial; font-size: 15px; line-height: 12pt;">Choose a cutoff date to split out a validation dataset</p>', unsafe_allow_html=True)
            df.sort_values(by='ds', ascending=True)
            cutoff= st.select_slider(label = "Select Cutoff Date", options = df['ds'], value = [min(df['ds'].head(13)),min(df['ds'].tail(13))])
            
            with st.expander('Model Parameters', expanded = True):
                freq = st.radio(label='Select frequency of the data (Default = Monthly)', options=['MS', 'D'], format_func = lambda x: {'MS': 'Monthly', 'D': 'Daily'}.get(x), index = 0)
                
                st.markdown('<p style="font-family:Arial; color:#7500C0; font-weight: bold; font-size: 15px; line-height: 15pt; font-variant: small-caps;">growth model</p>', unsafe_allow_html=True)
                growth = st.selectbox(label='Select your saturating forecast parameter to forecast',options=['linear','logistic']) 
                if growth == 'linear':
                    growth_settings= {'cap':1,
                                      'floor':0}
                    cap=1
                    floor=1
                    df['cap']=1
                    df['floor']=0
               
                st.markdown('<p style="font-family:Arial; color:#7500C0; font-weight: bold; font-size: 15px; line-height: 15pt; font-variant: small-caps;">confidence interval</p>', unsafe_allow_html=True)
                intervalw = st.number_input(label= '(Default = 95%): Range of values for the prediction. Options include 80, 85, 90, 95',min_value=.55, max_value = .95, step = .05, value = .95)
                
                
                if page == "Forecast": #this option is dymamic in the model training section
                    st.markdown('<p style="font-family:Arial; color:#7500C0; font-weight: bold; font-size: 15px; line-height: 15pt; font-variant: small-caps;">horizon</p>', unsafe_allow_html=True)
                    periods_input = st.slider('Select how many future periods (months) to forecast.',
                    min_value = 1, max_value = 365,value=12)
          
                # Changepoint 
                st.markdown('<p style="font-family:Arial; color:#7500C0; font-weight: bold; font-size: 15px; line-height: 15pt; font-variant: small-caps;">changepoints</p>', unsafe_allow_html=True)
                n_changepoints= st.number_input(label= 'Sets the ceiling on the number of potential trend changepoints detected in the first 80% of historical data.',min_value=1, max_value = 25, step = 1, value = 25)               
                changepoint_scale= st.text_input('Input changepoint prior to scale here:', value = 0.02)
                # st.text('Your changepoint prior to scale: %s' %(changepoint_scale)) #uncomment to preview selected changepoint
                    
                # Seasonality mode
                st.markdown('<p style="font-family:Arial; color:#7500C0; font-weight: bold; font-size: 15px; line-height: 15pt; font-variant: small-caps;">seasonality</p>', unsafe_allow_html=True)
                seasonality = st.radio(label='(Default = Multiplicative): Additive – Magnitude of the seasonality term remains constant. Multiplicative – Magnitude can change',options=['multiplicative', 'additive'], index = 0)
                yearly_seasonality_scale= st.text_input(label= 'Sets the Fourier order of the YEARLY seasonality term. If not applicable, input 0', value=5)
                weekly_seasonality_scale= st.text_input(label= 'Sets the Fourier order of the WEEKLY seasonality term. If not applicable, input 0', value=0)
                daily_seasonality_scale= st.text_input(label= 'Sets the Fourier order of the DAILY seasonality term. If not applicable, input 0', value=0)
                                
                #create parameter dataframe
                param_df = pd.DataFrame({"Growth": [growth],
                                         "Number of Changepoints": [n_changepoints] ,
                                         "Changepoint Scale": [changepoint_scale], 
                                         "Seasonality": [seasonality], 
                                         "Yearly Seasonality Scale": [yearly_seasonality_scale],
                                         "Monthly Seasonality Scale": [weekly_seasonality_scale],
                                         "Daily Seasonality Scale": [daily_seasonality_scale], 
                                         "Confidence Interval": [intervalw] })
                @st.cache
                def convert_df(param_df):
                    return param_df.to_csv().encode('utf-8')

                param = param_df.to_csv().encode('utf-8')
            

                st.download_button(label="Download Parameters as CSV",
                                   data=param,
                                   file_name='parameters.csv',
                                   mime='text/csv') 
            
# Create training and holdout datasets based on uploaded data            
            
            count_after = df[(df['ds'] > cutoff[1])].count() # this is for setting a dynamic horizon based on cutoff date
           
            df_holdout = df[(df['ds'] > cutoff[1])]
           
            df_train = df[(df['ds'].between(cutoff[0],cutoff[1]))]
                
           
            df = df_train
            
        viz, gap, compon=st.columns([1.2,0.05,1]) # Viz columns

        # Show training and holdout datasets within expanders

        with main1:               
            with st.expander('Training Dataset'):
                df_train.sort_values(by='ds', ascending=True)
                st.dataframe(df_train)
        with main1:
            with st.expander('Holdout Dataset'):
                df_holdout.sort_values(by='ds', ascending=True)
                st.dataframe(df_holdout)

        with main1:
            step3_text = '<p style="font-family:Arial; color:#7500C0; font-weight: bold; font-size: 25px; line-height: 15pt; font-variant: small-caps;">step two</p>'
            st.markdown(step3_text, unsafe_allow_html=True)
            
            if st.checkbox('Enable Live Refitting') or st.button("Generate Forecast",key="fit"): # Checking this allows the page to refresh on every parameter adjustement.
                if len(growth_settings)==2:
                    with st.spinner('Fitting the model..'):
                        m = model_init(df,
                                       growth,
                                       n_changepoints,
                                       changepoint_scale,
                                       yearly_seasonality_scale,
                                       weekly_seasonality_scale,
                                       daily_seasonality_scale,
                                       seasonality,
                                       intervalw)
                        future = m.make_future_dataframe(periods=count_after[1],freq=freq)
                        future['cap']=cap
                        future['floor']=floor
                        st.write("The model will produce forecast up to ", future['ds'].max())

                       
                        with st.spinner("Forecasting.."):
                            forecast = m.predict(future)
                        rslt_df = forecast[(forecast['ds'] > cutoff[1])]
                        rslt_df2 = forecast[(forecast['ds'].between(cutoff[0],cutoff[1]))]
                        with main1:    
                            st.metric("Training Data RMSE", int(rmse(df_train['y'], rslt_df2['yhat'])))
                        with main1:
                            st.metric("Holdout Data RMSE", int(rmse(df_holdout['y'], rslt_df['yhat'])))
                    
                
                with main2:                    
                    stepresults_text = '<p style="font-family:Arial; color:#7500C0; font-weight: bold; font-size: 25px; line-height: 15pt; font-variant: small-caps;">forecast results</p>'
                    st.markdown(stepresults_text, unsafe_allow_html=True)
                  
#################################                        
# Build Vega-lite Visualization #
#################################

# The training/holdout vega-lite viz is slightly more complicated. 
                    


                    df_holdout = df_holdout.rename(columns={'y': 'y_orig'}) # rename holdout columns
                    
                    df_holdout['ds'] = pd.to_datetime(df_holdout['ds']) # convert holdout date to dt
                    df_holdout['holdout'] = '2 - holdout' # identify holdout rows for dashed line
                    
                    
                    chart_data=pd.merge(forecast, m.history, how = 'left') # create intial chart data
                    
                    
                    
                    chart_data2=pd.merge(df_holdout, chart_data, how = 'right', on = 'ds') # join chart data to holdout dataset
                    chart_data_download = chart_data2 #this is for the downloadable version
                    chart_data2['holdout'] = np.where(chart_data2['holdout'] != '2 - holdout', "1 - training", "2 - holdout")
                    add_row = chart_data2[(chart_data2['ds'] == cutoff[1])] # this is to connect the solid and dashed lines in the chart
                    chart_data2=chart_data2.append(add_row, ignore_index=True)  # append duplicated row
                    chart_data2.loc[chart_data2.tail(1).index, 'holdout'] = '2 - holdout'  # remove holdout from duplicated row
                    
                    # call chart                  
                    st.vega_lite_chart(chart_data2, { "layer": [
                                                                    {
                                                                    'mark': {'type': 'line', 'tooltip': True},
                                                                    'width': 1200,
                                                                    'height': 600,
                                                                    'encoding': {
                                                                        'x': {'field': 'ds', 'type': 'temporal'},
                                                                        'y': {'field': 'yhat', 'type': 'quantitative'},
                                                                        "color": {"value": "#96968c"},
                                                                        "strokeDash": {"field": "holdout", "type": "nominal"},
                                                                                }                           
                                                                    },                                    
                                                                    {
                                                                    "mark": {"type": "area", "tooltip": True},
                                                                    "encoding": {
                                                                        'x': {'field': 'ds', 'type': 'temporal', 'title' : 'Date'},
                                                                        "y": {"aggregate": "min",
                                                                        "field": "yhat_lower",
                                                                        "color": {"value": "#111111"},
                                                                        'title' : ' USD ($)',
                                                                        "opacity": { "value": 0.08 }
                                                                        },
                                                                        "y2": {"aggregate": "max", "field": "yhat_upper"},
                                                                        "color": {"value": "#111111"},
                                                                    
                                                                        "opacity": { "value": 0.08 }
                                                                    }
                                                                    },
                                                                    {
                                                                    'mark': {'type': 'circle', 'tooltip': True},
                                                                    'width': 1200,
                                                                    'height': 600,
                                                                    'encoding': {
                                                                        'x': {'field': 'ds', 'type': 'temporal'},
                                                                        'y': {'field': 'y', 'type': 'quantitative'},
                                                                        "color": {"value": "#460073"},
                                                                        "size": {"value": 50},
                                                                        "opacity": { "value": 1 },
                                                                        
                                                                        }
                                                                    },
                                                                    {
                                                                    
                                                                    'mark': {'type': 'circle', 'tooltip': True},
                                                                    'width': 1200,
                                                                    'height': 600,
                                                                    'encoding': {
                                                                        'x': {'field': 'ds', 'type': 'temporal'},
                                                                        'y': {'field': 'y_orig', 'type': 'quantitative'},
                                                                        "color": {"value": "#460073"},
                                                                        "size": {"value": 50},
                                                                        "opacity": { "value": 1 },
                                                                        }              
                                                                    }]
                                                    })
            
                with main1:      
                    
                    with st.expander('Forecast Dataset'):
                        chart_data_dl = chart_data_download[['ds', 'y', 'y_orig', 'yhat', 'yhat_upper', 'yhat_lower']] # create results dataset
                        chart_data_dl.rename(columns={'ds': 'Date', 'y' : 'Training Actuals', 'y_orig' : 'Cutoff Actuals', 'yhat' : "Forecast", 'yhat_upper' : 'Upper Bound', 'yhat_lower' : 'Lower Bound'}, inplace=True)
                        chart_data_dl.sort_values(by='Date', inplace=True, ascending=False)
                        st.dataframe(chart_data_dl)
                        param_df = pd.DataFrame({"Growth": [growth],
                                         "Number of Changepoints": [n_changepoints] ,
                                         "Changepoint Scale": [changepoint_scale], 
                                         "Seasonality": [seasonality], 
                                         "Yearly Seasonality Scale": [yearly_seasonality_scale],
                                         "Monthly Seasonality Scale": [weekly_seasonality_scale],
                                         "Daily Seasonality Scale": [daily_seasonality_scale], 
                                         "Confidence Interval": [intervalw] })
                        st.dataframe(param_df)


                    @st.cache
                    def convert_df(forecast):
                         # IMPORTANT: Cache the conversion to prevent computation on every rerun
                        return chart_data_dl.to_csv().encode('utf-8')

                    

                    csv = convert_df(chart_data_dl)

                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='forecast.csv',
                        mime='text/csv')

                   

                    #with st.spinner("Loading.."):
                with main2:
                    st.markdown("#")
                    stepcomps_text = '<p style="font-family:Arial; color:#7500C0; font-weight: bold; font-size: 25px; line-height: 15pt; font-variant: small-caps;">model components</p>'
                    st.markdown(stepcomps_text, unsafe_allow_html=True)      

                    fig3 = m.plot_components(forecast)
                    st.write(fig3)     

                















##################################################
# 
#     
# def train_test_split(df, periods):
#     train_data = df.iloc[:len(df)-periods, :]
#     test_data  = df.iloc[len(df)-periods:, :]
#     return train_data,test_data

# def model_init(df,cps,yearly_seasonality,weekly_seasonality,daily_seasonality,seasonality_mode):
#     m = Prophet.Prophet(seasonality_mode=seasonality_mode,
#                 changepoint_prior_scale=cps,
#                 yearly_seasonality = yearly_seasonality,
#                 weekly_seasonality = weekly_seasonality,
#                 daily_seasonality = daily_seasonality
#                 )
#     m.fit(df)
#     return m

# def make_forecast(model, periods, freq):
#     future = model.make_future_dataframe(periods=periods, freq=str(freq))
#     forecast = model.predict(future)
#     return forecast

# def rmse(predictions, targets):
#     differences = predictions - targets                       
#     differences_squared = differences ** 2                    
#     mean_of_differences_squared = differences_squared.mean()  
#     rmse_val = np.sqrt(mean_of_differences_squared)           
#     return rmse_val 


# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write(df)

# periods = st.radio('Please select forecasting options:',
#                     [12,18]
#                     )
# st.write('You have to select option %s months.' %(int(periods)))

# sm = st.radio('Please select forecasting options:',
#                     ['multiplicative','additive'] 
#                     )
# st.write('You have to select option %s for seasonality mode.' %(sm))

# cps = st.text_input('Input changepoint prior to scale here:', value = 0.02) 
# st.write('Here is your input for changepoint prior to scale: %s' %(cps))

# yearly_season = st.text_input('Input yearly seasonality here:', value = 12) 
# st.write('Here is your input for yearly seasonality: %s' %(yearly_season))

# weekly_season = st.text_input('Input weekly seasonality here:', value = 0)
# st.write('Here is your input for monthly seasonality: %s' %(weekly_season)) 

# daily_season = st.text_input('Input daily seasonality here:', value = 0) 
# st.write('Here is your input for daily season: %s' %(daily_season))

# # df = read_data(dir_link)

# train_data,test_data = train_test_split(df,int(periods))
# model = model_init(train_data, float(cps), yearly_season,weekly_season,daily_season,sm)
# result = make_forecast(model, int(periods), 'MS' )
# your_rmse = rmse(result['yhat'].tail(12).to_numpy(),test_data['y'].to_numpy())

# st.text('Here is your RMSE for the forecast: ' + str(your_rmse))
# st.dataframe(result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))