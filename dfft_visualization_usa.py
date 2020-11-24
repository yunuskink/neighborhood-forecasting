import ipyleaflet
from branca.colormap import linear
from shapely.geometry import shape, Point
import numpy as np
import pandas as pd
import ipywidgets as widgets
import bqplot as bq
import dfft_forecasting as dfft_fcast
import dfft_io


def update_county_data(df_blockgroup, geoid_county):
    df_blockgroup_county = df_blockgroup[df_blockgroup['county_GEOID']==geoid_county]
    df_blockgroup_county.replace([np.inf, -np.inf], np.nan)
    df_blockgroup_county = df_blockgroup_county.dropna(inplace=False)
    geo_df_blockgroup, geojson_blockgroup = dfft_io.load_blockgroup_GIS(geoid_county)
    # Remove from shapefiles any block groups that are not present in df_blockgroup_county
    rows_feature = [feature['properties']['GEOID10'] for feature in geojson_blockgroup['features']]
    rows_df_blockgroup_county = [row['bg_GEOID'] for index, row in df_blockgroup_county.iterrows()]
    blockgroups_to_remove = np.setxor1d(rows_feature, rows_df_blockgroup_county, assume_unique=True)
    indices_to_remove = []
    for i in range(len(geojson_blockgroup['features'])):
        if geojson_blockgroup['features'][i]['id'] in blockgroups_to_remove:
            # if geojson_blockgroup['features'][i]['GEOID10'] in blockgroups_to_remove:
            indices_to_remove.append(i)
    for index in sorted(indices_to_remove, reverse=True):
        geojson_blockgroup['features'].pop(index)
    return geojson_blockgroup, df_blockgroup_county, geo_df_blockgroup


def build_forecasting_app(forecasting_df_filename = "./data/forecasting_parameters_df_Boris.csv"):
    """
    PLANS: A user need only run this lines of code and the user will be presented with
    an interactive map demonstrating both how compositions of cities changed from 2000-2010
    and how our forecasts matched with those observations for White/non-White, Black/non-Black
    and Hispanic/non-Hispanic populations.
    User workflow:
    First, user will need to use the search tool to place the marker within a county of interest.
    Second, user will be presented with an animated heatmap for the probability of observing a
    future composition given the initial composition for a neighborhood of 1000 persons
    Third, user can choose from a dropdown menu to see 9 separate entries for the 2000,2010 observed
    and 2010 forecasted mean for the selected county.
    Fourth, user will be able to click on a neighborhood of interest and see an animated plot for
    the dynamics of that specific neighborhood
    To explore a new county, the user need only use the search tool to find a new county.
    BACKEND:
    First, I need a function that is called when the search tool is used. A county should then be
    selected and the shapefile, Headache, optimum delta_V, and optimum delta_T should be loaded
    from file or calculated. This will be a dataframe.
    Second,

    App widgets...
    1) race_selector: lets user choose whether they want Black/non-Black,White/non-White,Hispanic/non-Hispanic
    2) decade_selector: lets user choose whteher to see just a single decade or the change between two decades
    3) SearchControl: Lets users find new location
    4) marker: Used to collect the location of a given county for analysis
    5) year_slider: Lets users forecast further into the future and intervening years

    :return:
    """
    initial_position = (41.7406, -87.7093) #Somewhere in Cook county
    years = np.linspace(2000, 2019, 20, 'Int')

    ###### LOADING DATA ################
    df_blockgroup = pd.read_pickle("./data/blockgroup_longitudinal_data")
    df_blockgroup = dfft_io.add_columns_to_blockgroup_df(df_blockgroup) #Calculate columns like fractional change
    geo_df_county = dfft_io.load_county_GIS()
    forecasting_parameters_df = dfft_io.load_dataframe_with_H_from_csv("./data/forecasting_parameters_df_Boris.csv")

    ######### FORECAST FOR INITIAL NEIGHBORHOOD ################
    # Get id for initial county
    geoid_county = geo_df_county.loc[geo_df_county.contains(Point(initial_position[1], initial_position[0]))]
    geoid_county = geoid_county.iloc[0]['GEOID']

    # Find the blockgroup that matches the location
    geo_df_blockgroup, geojson_blockgroup = dfft_io.load_blockgroup_GIS(geoid_county)
    geoid_blockgroup = geo_df_blockgroup.loc[
        geo_df_blockgroup.contains(Point(initial_position[1], initial_position[0]))]
    geoid_blockgroup = geoid_blockgroup.iloc[0]['GEOID10']
    data_selected_blockgroup = df_blockgroup[df_blockgroup['bg_GEOID'] == geoid_blockgroup]
    forecasting_parameters = forecasting_parameters_df[(forecasting_parameters_df["county"] == 1001) &
                                                         (forecasting_parameters_df["decade_initial"] == 2000) &
                                                       (forecasting_parameters_df["decade_final"] == 2010)
                                                       ]
    forecasts = dfft_fcast.generate_forecasts_selected_blockgroup(forecasting_parameters, data_selected_blockgroup,
                                                                  years)
    ######## WIDGETS ############
    # race_selector = widgets.Dropdown(options=('black', 'hispanic', 'white'),
    race_selector = widgets.ToggleButtons(options=(['black','hispanic','white']),
                                          description = 'Race/Ethnicity:',
                                     # layout=widgets.Layout(width='20%'),
                                          style={'description_width': 'initial'})
    decade_selector = widgets.Dropdown(options=('1990', '2000', '2010', '1990->2000', '1990->2010', '2000->2010'),
                                       description='Decade Layer:',
                                       # layout=widgets.Layout(width='20% '),
                                       style={'description_width': 'initial'})
    validation_forecast_selector = widgets.ToggleButtons(options=(['Validation on 2010','Forecast beyond 2010']),
                                         layout=widgets.Layout(width='auto'))
    file = open("./red_blue_colorbar.png", "rb");red_blue_colorbar_img = file.read()
    file = open("./purples_colorbar.png", "rb");purples_colorbar_img = file.read()
    colorbar_image = widgets.Image(
        value=purples_colorbar_img,
        format='png',
        width=300,
        height=400,
    )
    validation_forecast_selector.value = 'Validation on 2010'
    race_selector.value = 'white'
    decade_selector.value = '2000'
    year_slider = widgets.IntSlider(min=2000,
                                    max=2010,
                                    step=1,
                                    description='Year',
                                    orientation='horizontal')
    header = widgets.HTML("<h1>Forecasts of neighborhood compositions</h1>", layout=widgets.Layout(height='auto'))
    header.style.text_align = 'center'
    out = widgets.HTML(
        value='',
        layout=widgets.Layout(width='auto', height='auto')
    )

    def get_column_name():
        nonlocal decade_selector, race_selector
        if len(decade_selector.value)==4: #Choosing just a decade, not the dynamic change
            name = "fraction_" + race_selector.value + "_" + decade_selector.value[2:]
        else:
            name = "delta_fraction_" + race_selector.value + "_" + decade_selector.value[2:4] + \
                "_" + decade_selector.value[8:10]
        return name

    ####### INITIALIZE MAP ##################################
    geojson_blockgroup, df_blockgroup_county, geo_df_blockgroup = update_county_data(df_blockgroup, geoid_county)
    m = ipyleaflet.Map(center=initial_position, zoom=10)
    # Choropleth layer for data in column_name
    marker = ipyleaflet.Marker(location=initial_position, draggable=True)
    m.add_control(ipyleaflet.SearchControl(
        position="topleft",
        url='https://nominatim.openstreetmap.org/search?format=json&q={s}',
        zoom=10,
        marker=marker
    ))
    m.add_layer(marker)

    base_layers = m.layers
    column_name = get_column_name()
    layer_blockgroup = ipyleaflet.Choropleth(
        geo_data=geojson_blockgroup,
        choro_data=dict(zip(df_blockgroup_county[:]['bg_GEOID'], df_blockgroup_county[:][column_name])),
        colormap=linear.Purples_05,
        stroke='False',
        value_min=0,
        value_max=1,
        style={'fillOpacity': 0.8, 'weight': 0})
    m.add_layer(layer_blockgroup)
    m.add_control(decade_selector)

    # Now initialize the line plots
    lin_x = bq.LinearScale()
    lin_y = bq.LinearScale()
    lin_x.min = 0.0
    lin_x.max = 1
    ax_x_bg_forecast = bq.Axis(label='Composition, n', scale=lin_x)
    ax_y_bg_forecast = bq.Axis(label='Probability, P', scale=lin_y, orientation='vertical',label_offset = '70px')
    line_probability = bq.Lines(x=np.linspace(0, 1, len(forecasts[race_selector.value][year_slider.value - year_slider.min, :])),
                           y=forecasts[race_selector.value][year_slider.value - year_slider.min, :],
                           scales={'x': lin_x, 'y': lin_y})
    max_prob = max(forecasts[race_selector.value][year_slider.value - year_slider.min, :])
    n_1990 = data_selected_blockgroup.iloc[0]["fraction_"+race_selector.value+"_90"]
    n_2000 = data_selected_blockgroup.iloc[0]["fraction_"+race_selector.value+"_00"]
    n_2010 = data_selected_blockgroup.iloc[0]["fraction_"+race_selector.value+"_10"]
    line_1990 = bq.Lines(x=[n_1990, n_1990], y=[0, 1], scales={'x': lin_x, 'y': lin_y},preserve_domain={'x': True, 'y': False},colors = ['black'])
    line_2000 = bq.Lines(x=[n_2000, n_2000], y=[0, 1], scales={'x': lin_x, 'y': lin_y},preserve_domain={'x': True, 'y': False},colors = ['black'])
    line_2010 = bq.Lines(x=[n_2010, n_2010], y=[0, 1], scales={'x': lin_x, 'y': lin_y},preserve_domain={'x': True, 'y': False},colors = ['black'])
    lbl_1990 = bq.Label(x=[n_1990], y=[0.25*max_prob], scales={'x': lin_x, 'y': lin_y}, text=['1990'],align = "middle",colors = ['black'])
    lbl_2000 = bq.Label(x=[n_2000], y=[0.5*max_prob], scales={'x': lin_x, 'y': lin_y}, text=['2000'], align="middle",colors = ['black'])
    lbl_2010 = bq.Label(x=[n_2010], y=[0.75*max_prob], scales={'x': lin_x, 'y': lin_y}, text=['2010'], align="middle",colors = ['black'])

    margin_fig = dict(left=100, top=50, bottom=50, right=100)
    fig_bg_forecast = bq.Figure(axes=[ax_x_bg_forecast, ax_y_bg_forecast],
                                marks=[line_probability,line_1990,line_2000,line_2010,lbl_1990,lbl_2000,lbl_2010],
                    fig_margin=margin_fig)

    # TODO: Put dropdown menus on top of map
    # app = widgets.AppLayout(left_sidebar=m,
    #                         header=header,
    #                         # right_sidebar=widgets.VBox([fig_bg_forecast, year_slider, plot_type_toggle]),
    #                         center = widgets.VBox([fig_bg_forecast, year_slider]),
    #                         # right_sidebar= fig_county_forecast,
    #                         footer=out,
    #                         pane_widths=['80px', 1, 1],
    #                         pane_heights=['80px', 4, 1],
    #                         height='600px',
    #                         grid_gap="30px")

    app = widgets.AppLayout(center=widgets.VBox([decade_selector,m,colorbar_image]),
                            header=widgets.VBox([header,race_selector]),
                            # left_sidebar=widgets.VBox([widgets.Label("Race/Ethnicity:"),
                            #                            race_selector]),
                            #                            # widgets.Label("Year:"),
                            #                            # decade_selector]),
                            # left_sidebar=widgets.VBox([widgets.Label("Race/Ethnicity:"),
                            #                            race_selector,
                            #                            widgets.Label("Year:"),
                            #                            decade_selector]),
                            right_sidebar=widgets.VBox([validation_forecast_selector,year_slider,fig_bg_forecast]),
                            # footer=out,
                            pane_widths=['200px', '200px', '300px'],
                            pane_heights=['100px', 4, 1],
                            height='700px',
                            grid_gap="30px")

    #TODO: Add in a interaction for clicking on the map?
    ################## INTERACTIONS ##############################
    # def handle_interaction(**kwargs):
    #     if kwargs['type'] == 'click':
    #         generate_temp_series(*kwargs['coordinates'])
    #         msg = '%s Selected coordinates: %s, Temp: %d C Precipitation: %d mm\n' % (
    #             kwargs['coordinates'], random.randint(-20, 20), random.randint(0, 100))
    #         out.value = add_log(msg)
    #
    # m.on_interaction(handle_interaction)

    def on_location_changed(event):
        print(event)
        location = event['new']
        geoid_county_new = geo_df_county.loc[geo_df_county.contains(Point(location[1], location[0]))]
        geoid_county_new = geoid_county_new.iloc[0]['GEOID']
        nonlocal geo_df_blockgroup, geojson_blockgroup, df_blockgroup, base_layers
        nonlocal line_probability
        nonlocal year_slider, forecasts, geoid_county
        nonlocal geoid_blockgroup, data_selected_blockgroup, forecasting_parameters,n_1990,n_2000,n_2010
        nonlocal initial_position, column_name, years
        nonlocal df_blockgroup, df_blockgroup_county, layer_blockgroup, colorbar_image
        # nonlocal fig_county_forecast, county_heatmap_plots
        if geoid_county != geoid_county_new:
            geoid_county = geoid_county_new
            geojson_blockgroup, df_blockgroup_county, geo_df_blockgroup = update_county_data(df_blockgroup, geoid_county)
            column_name = get_column_name()
            # county_heatmap_plots = generate_heatmap_plots(df_blockgroup_county, forecasting_parameters, years)
            # fig_county_forecast = county_heatmap_plots[race_selector.value][year_slider.value]
            if column_name[0:5] == "delta":
                value_min = -0.2
                value_max = 0.2
                cmap = linear.RdBu_09
                colorbar_image.value = red_blue_colorbar_img
            else:
                value_min = 0
                value_max = 1
                cmap = linear.Purples_05
                colorbar_image.value = purples_colorbar_img
            layer_blockgroup = ipyleaflet.Choropleth(
                geo_data=geojson_blockgroup,
                choro_data=dict(zip(df_blockgroup_county[:]['bg_GEOID'], df_blockgroup_county[:][column_name])),
                colormap=cmap,
                stroke='False',
                value_min=value_min,
                value_max=value_max,
                style={'fillOpacity': 0.8, 'weight': 0})
            # all_layers = all_layers + (layer_blockgroup,)
            m.layers = base_layers
            m.add_layer(layer_blockgroup)

            # TODO: Plot the county wide forecast
        geoid_blockgroup = geo_df_blockgroup.loc[
            geo_df_blockgroup.contains(Point(location[1], location[0]))]
        geoid_blockgroup = geoid_blockgroup.iloc[0]['GEOID10']
        data_selected_blockgroup = df_blockgroup[df_blockgroup['bg_GEOID'] == geoid_blockgroup]
        n_1990 = data_selected_blockgroup.iloc[0]["fraction_" + race_selector.value + "_90"]
        n_2000 = data_selected_blockgroup.iloc[0]["fraction_" + race_selector.value + "_00"]
        n_2010 = data_selected_blockgroup.iloc[0]["fraction_" + race_selector.value + "_10"]
        forecasting_parameters = forecasting_parameters_df[(forecasting_parameters_df["county"] == 1001) &
                                                           (forecasting_parameters_df["decade_initial"] == 2000) &
                                                           (forecasting_parameters_df["decade_final"] == 2010)
                                                           ]
        forecasts = dfft_fcast.generate_forecasts_selected_blockgroup(forecasting_parameters, data_selected_blockgroup, years)
        # lines_black.y = forecasts["black"][year_slider.value-year_slider.min]
        # lines_hispanic.y = forecasts["hispanic"][year_slider.value - year_slider.min]
        line_probability.y = forecasts["white"][year_slider.value - year_slider.min]
        line_probability.x = np.linspace(0, 1, len(forecasts[race_selector.value][year_slider.value - year_slider.min, :]))
        max_prob = max(forecasts["white"][year_slider.value - year_slider.min])
        line_1990.x = [n_1990,n_1990];line_2000.x = [n_2000,n_2000];line_2010.x = [n_2010,n_2010]
        line_1990.y = [0, max_prob];line_2000.y = [0, max_prob];line_2010.y = [0, max_prob]
        lbl_1990.x = [n_1990];lbl_2000.x = [n_2000];lbl_2010.x = [n_2010]
        lbl_1990.y = [0.25*max_prob];lbl_2000.y = [0.5*max_prob];lbl_2010.y = [0.75*max_prob]
        # vline_white.x = [1, 1]
        return
        # Generate forecasts of the years 2001 to 2019

    def on_race_decade_change(change):
        # print(type(change))
        if change['type'] == 'change' and change['name'] == 'value':
            nonlocal layer_blockgroup, df_blockgroup_county, geojson_blockgroup
            col = get_column_name()
            if col[0:5] == "delta":
                value_min = -0.2
                value_max = 0.2
                cmap = linear.RdBu_09
                colorbar_image.value = red_blue_colorbar_img
            else:
                value_min = 0
                value_max = 1
                cmap = linear.Purples_05
                colorbar_image.value = purples_colorbar_img
            layer_blockgroup = ipyleaflet.Choropleth(
                geo_data=geojson_blockgroup,
                choro_data=dict(zip(df_blockgroup_county[:]['bg_GEOID'], df_blockgroup_county[:][col])),
                colormap=cmap,
                stroke='False',
                value_min=value_min,
                value_max=value_max,
                style={'fillOpacity': 0.8, 'weight': 0})
            # all_layers = all_layers + (layer_blockgroup,)
            m.layers = base_layers
            m.add_layer(layer_blockgroup)

    def update_year(change):
        if change['type'] == 'change' and change['name'] == 'value':
            line_probability.y = forecasts[race_selector.value][year_slider.value - year_slider.min]
            line_probability.x = np.linspace(0, 1, len(forecasts[race_selector.value][year_slider.value - year_slider.min, :]))
            max_prob = max(forecasts["white"][year_slider.value - year_slider.min])
            line_1990.x = [n_1990, n_1990];line_2000.x = [n_2000, n_2000];line_2010.x = [n_2010, n_2010]
            line_1990.y = [0, max_prob];line_2000.y = [0, max_prob];line_2010.y = [0, max_prob]
            lbl_1990.y = [0.25 * max_prob];lbl_2000.y = [0.5 * max_prob];lbl_2010.y = [0.75 * max_prob]

    # def update_plot_type(change):
    #     print(change)
    #     # if change['type'] == 'change' and change['name'] == 'value':
    #     nonlocal app
    #     if change['new'] == "Neighborhood":
    #         right_sidebar = widgets.VBox([fig_bg_forecast, year_slider, plot_type_toggle])
    #     elif change['new'] == "County":
    #         right_sidebar = widgets.VBox([fig_county_forecast, year_slider, plot_type_toggle])
    #     else:
    #         right_sidebar = widgets.VBox([fig_bg_forecast, year_slider, plot_type_toggle])
    #
    #     app = widgets.AppLayout(center=m,
    #                             header=header,
    #                             left_sidebar=widgets.VBox([widgets.Label("Race/Ethnicity:"),
    #                                                        race_selector,
    #                                                        widgets.Label("Year:"),
    #                                                        decade_selector]),
    #                             right_sidebar=right_sidebar,
    #                             footer=out,
    #                             pane_widths=['80px', 1, 1],
    #                             pane_heights=['80px', 4, 1],
    #                             height='600px',
    #                             grid_gap="30px")
    #

    ##### INTERACTION FUNCTION CALLS ##############
    #TODO: Add in call to on_location_changed when use search tool for new place

    #TODO: Add in to "on_race_decade_change" code to also update the lines
    race_selector.observe(on_race_decade_change)
    decade_selector.observe(on_race_decade_change)
    year_slider.observe(update_year)
    marker.observe(on_location_changed, 'location')
    # plot_type_toggle.observe(update_plot_type)

    return app



#
# def generate_forecasts_selected_blockgroup(H_dict, steps_per_year_all_races, mus, data_selected_blockgroup, years):
#     """
#     Generate forecasts for the probability of observing a given fraction of persons for binary White, Black, and Hispanic
#     classifications
#     :param H_functions:
#     :param steps_per_year_all_races:
#     :param mus:
#     :param data_selected_blockgroup:
#     :return:
#     """
#     if years[0] == 2000: # Leaving this in for now to allow flexibility later to produce forecasts from 1990 instead
#         total = data_selected_blockgroup['total_00']
#     # elif years[0] == 1990:
#     #     total = data_selected_blockgroup['total_90']
#     transition_matrices = {
#         'black': dfft_fcast.build_transition_matrix(dfft_fcast.shift_H_by_mu(H_dict['black_00'], mus['black_00_10']), total),
#         'hispanic': dfft_fcast.build_transition_matrix(dfft_fcast.shift_H_by_mu(H_dict['hispanic_00'], mus['hispanic_00_10']), total),
#         'white': dfft_fcast.build_transition_matrix(dfft_fcast.shift_H_by_mu(H_dict['white_00'], mus['white_00_10']), total)}
#
#     def forecast_from_transition_matrix(transition_matrix, N_initial, number_years, tot, steps_per_year):
#         state_vector = np.zeros((tot + 1, 1))
#         state_vector[N_initial] = 1
#         transition_matrix = np.linalg.matrix_power(transition_matrix, round(steps_per_year * tot))
#         forecast = np.zeros(tot + 1, number_years)
#         forecast[:, 0] = state_vector
#         for i in range(number_years - 1):
#             state_vector = np.matmul(transition_matrix, state_vector)
#             forecast[:, i + 1] = state_vector
#         return forecast
#
#     forecasts = {'black': forecast_from_transition_matrix(transition_matrices['black'],
#                                                           round(data_selected_blockgroup['black_00']),
#                                                           len(years), total, steps_per_year_all_races['black']),
#                  'hispanic': forecast_from_transition_matrix(transition_matrices['hispanic'],
#                                                              round(data_selected_blockgroup['hispanic_00']),
#                                                              len(years), total, steps_per_year_all_races['hispanic']),
#                  'white': forecast_from_transition_matrix(transition_matrices['white'],
#                                                           round(data_selected_blockgroup['white_00']),
#                                                           len(years), total, steps_per_year_all_races['white'])
#                  }
#     return forecasts
#
#
# rows = []
# h = np.load('dict_h.npy', allow_pickle = True).item()
#
# # appending rows
# for decade_key, h_decade in h.items():
#     for race_key,h_race in h_decade.items():
#         for county_key,h_county in h_race.items():
#             data_row = data['Student']
#             time = data['Name']
#             row = {"decade": decade_key, "race": race_key, "county": county_key, "H": h_county}
#             rows.append(row)
#             print(county_key)
#
# df = pd.DataFrame(rows)


# [
# "Paired", "Set3", "Pastel1", "Set1",
# "Greys", "Greens", "Reds", "Purples", "Oranges", "Blues",
# "YlOrRd", "YlOrBr", "YlGnBu", "YlGn", "RdPu",
# "PuRd", "PuBuGn", "PuBu", "OrRd", "GnBu", "BuPu",
# "BuGn", "BrBG", "PiYG", "PRGn", "PuOr", "RdBu", "RdGy",
# "RdYlBu", "RdYlGn", "Spectral"
# ]