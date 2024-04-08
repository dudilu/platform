import streamlit as st
import pandas as pd
import os
from datetime import datetime
from streamlit_option_menu import option_menu
import base64
import numpy as np
import altair as alt
import plotly.express as px


columns = ['lot', 'wafer', 'IC X', 'IC Y', 'Device', 'mode', 'mean', 'std']

concatenated_results_pca = pd.DataFrame(columns=columns)

def correlation_plot(df, x_col, y_col):
    hover_data = [df['ult']]

    if 'ult' in df['ult'].values:
        hover_data.append(df.apply(lambda row: row['ult'] if row['ult'] == 'ult' else None, axis=1))

    h2_title = f"<h2 style='color:#74B6FF'>Correlation Plot: {x_col} vs. {y_col}</h2>"
    fig = px.scatter(df, x=x_col, y=y_col, color='lot',
                     labels={x_col: x_col, y_col: y_col},
                     hover_data=hover_data,
                     height=800)


    st.markdown(h2_title, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)


def recode_value(index):
    recode_mapping = {
        0: "pulvt",
        1: "plvt",
        2: "psvt",
        3: "nulvt",
        4: "nlvt",
        5: "nsvt",
        6: "pulvtll",
        7: "plvtll",
        8: "pelvt",
        9: "nulvtll",
        10: "nlvtll",
        11: "nelvt"
    }

    if index in recode_mapping:
        return recode_mapping[index]
    else:
        return "Invalid"
def add_all(list_wo_all):
    return np.append('All', list_wo_all)
def only_relevant(data,from_select,name_from,name_to):
    return data[data[name_from].isin(from_select)][name_to].unique()

# folder_path = "C:/Users/DudiLubton/PycharmProjects/pythonProject/platform/agg1"
# folder_path = "C:/Users/DudiLubton/PycharmProjects/pythonProject/platform/agg1"
# folder_path1 = "C:/Users/DudiLubton/PycharmProjects/pythonProject/platform/agg1/iddq"

folder_path = "https://raw.githubusercontent.com/dudilu/platform/main"
folder_path1 = "https://raw.githubusercontent.com/dudilu/platform/main/corr"
def plot_chart(color_by):
    chart = alt.Chart(data_to_use).mark_circle(size=60).encode(
        x=x_encoding,
        y=alt.Y('mean', scale=alt.Scale(domain=[data_to_use['lsl'].mean(), data_to_use['hsl'].mean()]), axis=alt.Axis(title='Cycle Time (nS)')),
        color=alt.Color(color_by + ':N'),
        tooltip=[x_encoding, 'mean', color_by]
    ).interactive()
    lsl_mean_line = alt.Chart(pd.DataFrame({'mean_lsl': [data_to_use['lsl'].mean()]})).mark_rule(color='red').encode(y='mean_lsl:Q')

    hsl_mean_line = alt.Chart(pd.DataFrame({'mean_hsl': [data_to_use['hsl'].mean()]})).mark_rule(color='red').encode(y='mean_hsl:Q')
    chart_with_lines = chart + lsl_mean_line + hsl_mean_line

    st.altair_chart(chart_with_lines, use_container_width=True)


def initialize_session_state():
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime.now().date() - pd.DateOffset(days=30)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.now().date()

st.set_page_config(page_title="Engineer Application", layout='wide', initial_sidebar_state="auto")



config_data = pd.read_csv('https://raw.githubusercontent.com/dudilu/platform/main/Date_config.csv', parse_dates=["Date"])
alerts = pd.read_csv('https://raw.githubusercontent.com/dudilu/platform/main/alerts.csv')


config_data["Date"] = pd.to_datetime(config_data["Date"], format='%d/%m/%Y')
unique_dates = config_data["Date"].dt.date.unique()
unique_lots = config_data['Filename'].unique()

initialize_session_state()

with st.sidebar:
    selected = option_menu("Engineer Application", ['📁 Upload file', '🏠 Control Panel', '📊 Process', '📈 Correlations'], menu_icon="cast")

    start_date = st.sidebar.date_input(
        "Select Start Date",
        min_value=min(unique_dates),
        max_value=max(unique_dates),
        value=st.session_state.start_date)

    end_date = st.sidebar.date_input(
        "Select End Date",
        min_value=min(unique_dates),
        max_value=max(unique_dates),
        value=st.session_state.end_date)

    selected_lots = st.sidebar.multiselect(
        "Select Lots",
        options=unique_lots,
        default=[])

    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())

    query_button = st.sidebar.button('Query')

if query_button:
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    selected_files_by_date = config_data[(config_data["Date"] >= start_date) & (config_data["Date"] < end_date)]["Filename"].tolist()
    selected_files_by_lot = config_data[config_data["Filename"].isin(selected_lots)]["Filename"].tolist()
    selected_files = list(set(selected_files_by_date) | set(selected_files_by_lot))
    st.session_state.selected_files = selected_files

if selected == "🏠 Control Panel":
    rows = [st.columns(1), st.columns(1)]
    with rows[0][0]:
        st.markdown(f'<h1 style="color:#74B6FF;">Alerts</h1>', unsafe_allow_html=True)
        st.write("")
        st.write("")
    with rows[1][0]:
        st.dataframe(alerts, use_container_width=True, hide_index=True)


if selected == "📈 Correlations":
    if st.session_state.selected_files:
        #data = pd.concat([pd.read_csv(os.path.join(folder_path1, file)) for file in st.session_state.selected_files], ignore_index=True)
        data = pd.concat([pd.read_csv(folder_path1 + '/' + file) for file in st.session_state.selected_files], ignore_index=True)

        column_options = [
            'iddq meas', 'nelvt_leak', 'nelvt_ref', 'nlvt_leak', 'nlvt_ref', 'nlvtll_leak', 'nlvtll_ref',
            'nsvt_leak', 'nsvt_ref', 'nulvt_leak', 'nulvt_ref', 'nulvtll_leak', 'nulvtll_ref',
            'pelvt_leak', 'pelvt_ref', 'plvt_leak', 'plvt_ref', 'plvtll_leak', 'plvtll_ref',
            'psvt_leak', 'psvt_ref', 'pulvt_leak', 'pulvt_ref', 'pulvtll_leak', 'pulvtll_ref', 'iddq_est']

        default_x_col = 'iddq meas'
        default_y_col = 'iddq_est'


        rows = [st.columns(1), st.columns(5), st.columns(1), st.columns(1)]



        with rows[1][0]:

            lots_list = data['lot'].unique()
            lots_list_all = add_all(lots_list)
            selected_lots = st.multiselect("Lot", options=lots_list_all, default='All' if len(lots_list_all) > 0 else None)

            if "All" in selected_lots:
                selected_lots = lots_list

        with rows[1][1]:
            wafers_list = only_relevant(data, selected_lots, 'lot', 'wafer')
            wafers_list_all = add_all(wafers_list)

            selected_wafer = st.multiselect('wafer', options=wafers_list_all, default='All')

            if "All" in selected_wafer:
                selected_wafer = wafers_list
            selected_wafer = np.array(selected_wafer, dtype=np.int64)

        with rows[1][2]:
            ults_list = only_relevant(data, selected_wafer, 'wafer', 'ult')

            ults_list_all = add_all(ults_list)

            selected_ult = st.multiselect('ult', options=ults_list_all, default='All')

            if "All" in selected_ult:
                selected_ult = ults_list

        data = data[
            data['lot'].isin(selected_lots) & data['ult'].isin(selected_ult) & data['wafer'].isin(selected_wafer)]


        with rows[0][0]:
            st.markdown(f'<h1 style="color:#74B6FF;">Correlations</h1>', unsafe_allow_html=True)

        with rows[1][3]:
            x_col = st.selectbox("", options=column_options,index=column_options.index(default_x_col))

        with rows[1][4]:
            y_col = st.selectbox("", options=column_options,
                                 index=column_options.index(default_y_col))
        with rows[2][0]:
            correlation_plot(data, x_col, y_col)

        with rows[3][0]:
            if st.button("Export Data"):
                st.write("Downloading data...")
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download Filtered Data as CSV</a>'
                st.markdown(href, unsafe_allow_html=True)


if selected == "📊 Process":
    if st.session_state.selected_files:
        #data = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in st.session_state.selected_files], ignore_index=True)
        data = pd.concat([pd.read_csv(folder_path  + '/' + file) for file in st.session_state.selected_files], ignore_index=True)

        rows = [st.columns(1), st.columns(4), st.columns(4), st.columns(1), st.columns(1)]

        with rows[0][0]:
            st.markdown(f'<h1 style="color:#74B6FF;">Statistical Process Control</h1>', unsafe_allow_html=True)


        with rows[1][0]:

            lots_list = data['lot'].unique()
            lots_list_all = add_all(lots_list)
            device_list = data['Device'].unique()
            mode_list = data['mode'].unique()
            selected_lots = st.multiselect("Lot", options=lots_list_all, default='All' if len(lots_list_all) > 0 else None)

            if "All" in selected_lots:
                selected_lots = lots_list

        with rows[1][1]:
            wafers_list = only_relevant(data, selected_lots, 'lot', 'wafer')
            wafers_list_all = add_all(wafers_list)

            selected_wafer = st.multiselect('wafer', options=wafers_list_all, default='All')

            if "All" in selected_wafer:
                selected_wafer = wafers_list
            selected_wafer = np.array(selected_wafer, dtype=np.int64)

        with rows[1][2]:
            ults_list = only_relevant(data, selected_wafer, 'wafer', 'ult')

            ults_list_all = add_all(ults_list)

            selected_ult = st.multiselect('ult', options=ults_list_all, default='All')

            if "All" in selected_ult:
                selected_ult = ults_list

        with rows[2][0]:
            selected_device = [st.selectbox('Device', options=device_list, index=0 if len(device_list) > 0 else None)]
        with rows[2][1]:
            selected_mode = [st.selectbox('Mode', options=mode_list, index=0 if len(mode_list) > 0 else None)]

        with rows[1][3]:
            test_stages_list = only_relevant(data, selected_lots, 'lot', 'Test Stage')
            test_stages_list_all = add_all(test_stages_list)

            selected_test = st.multiselect('Test Stage', options=test_stages_list_all,default=test_stages_list[0] if len(test_stages_list) > 0 else None)

            if "All" in selected_test:
                selected_test = test_stages_list

        filtered_data = data[
            data['lot'].isin(selected_lots) & data['Test Stage'].isin(selected_test) & data['ult'].isin(selected_ult) &
            data['wafer'].isin(selected_wafer) & data['Device'].isin(selected_device) & data['mode'].isin(selected_mode)]
        filtered_data1 = data[
            data['lot'].isin(selected_lots) & data['Test Stage'].isin(selected_test) & data['ult'].isin(selected_ult) &
            data['wafer'].isin(selected_wafer) & data['Device'].isin(selected_device) & data['mode'].isin(selected_mode)]

        filtered_data1 = filtered_data1.groupby(['lot', 'wafer', 'lsl', 'hsl'])['mean'].mean().reset_index()
        filtered_data1['lot_wafer'] = filtered_data1['lot'] + '_' + filtered_data1['wafer'].astype(str)

        with rows[2][3]:
            data_option = st.radio("Choose level:", ("chip level", "wafer level"))

        if data_option == "chip level":
            data_to_use = filtered_data
            x_encoding = 'ult'
        else:
            data_to_use = filtered_data1
            x_encoding = 'lot_wafer'

        with rows[2][2]:
            color_by_option = st.selectbox(
                "Color By:",
                ['lot', 'wafer'],
                index=0
            )



        st.write("")
        st.write("")
        with rows[3][0]:
            st.markdown(f'<h2 style="color:#74B6FF;">Variability Chart</h2>', unsafe_allow_html=True)
            plot_chart(color_by_option)

        with rows[4][0]:
            if st.button("Export Data"):
                st.write("Downloading data...")
                csv = data_to_use.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download Filtered Data as CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        st.write("No data available.")

if selected == "📁 Upload file":

    h2_upload_title = "<h2 style='color:#74B6FF'>Upload Test File</h2>"
    st.markdown(h2_upload_title, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload test files", type=['csv'], accept_multiple_files=True)

    if uploaded_files:

        progress_bar = st.progress(0)
        rows_processed = 0

        for uploaded_file in uploaded_files:

            readout_file = pd.read_csv(uploaded_file)

            for ro_ind in range(0, len(readout_file)):
                data = []
                s = 150
                ro_hex = readout_file.loc[ro_ind, 'IC Readout']

                ro_bin = bin(int(ro_hex, 16))[2:].zfill(len(ro_hex) * 4)

                for i in range(0, len(ro_bin)):
                    type_bin = ro_bin[s : s + 8]
                    type_dec = int(type_bin, 2)

                    length_bin = ro_bin[s + 8 : s + 8 + 12]
                    length_dec = int(length_bin, 2)

                    value_bin = ro_bin[s + 8 + 12 : s + 8 + 12 + length_dec]

                    data.append({'type': type_dec, 'length': length_dec, 'value_bin': value_bin})

                    s = s + 8 + 12 + length_dec
                    if type_dec == 11:
                        break

                df = pd.DataFrame(data)

                indices_9 = df.index[df['type'] == 9].tolist()
                indices_9.append(len(df))

                results_data = []
                for i in range(len(indices_9) - 1):
                    start_index = indices_9[i]
                    end_index = indices_9[i + 1]
                    sub_df = df.iloc[start_index:end_index]
                    sub_df = sub_df.reset_index(drop=True)

                    for j in range(1, len(sub_df)):
                        if sub_df.loc[j, 'type'] == 27:
                            for x in range(0, int((sub_df.loc[j, 'length'] - 49) / 5)):
                                results_data.append({
                                    'unit': int(sub_df.loc[0, 'value_bin'], 2),
                                    'Device': x,
                                    'value': int(sub_df.loc[j, 'value_bin'][49 + 5 * x: 49 + (x + 1) * 5], 2),
                                    'agent': 'ma'
                                })

                        if sub_df.loc[j, 'type'] == 7:
                            mode_ind = 0
                            for x in range(0, int((sub_df.loc[j, 'length'] - 10) / 10)):
                                results_data.append({
                                    'unit': int(sub_df.loc[0, 'value_bin'], 2),
                                    'Device': x // 4,
                                    'value': int(sub_df.loc[j, 'value_bin'][10 + 10 * x: 10 + (x + 1) * 10], 2),
                                    'agent': 'pca',
                                })

                                results_data[-1]['Device'] = recode_value(results_data[-1]['Device'])

                results = pd.DataFrame(results_data, columns=['unit', 'Device', 'value', 'agent'])
                results_pca = results[results['agent'] == 'pca']
                results_ma = results[results['agent'] == 'ma']

                results_pca['mode'] = ['ref' if i % 4 < 2 else 'leak' for i in range(len(results_pca))]
                results_pca = results_pca.reset_index(drop=True)

                for i in range(0, len(results_pca), 2):
                    results_pca.loc[i, 'cycle time'] = 46 * results_pca.loc[i, 'value'] / results_pca.loc[i + 1, 'value']

                results_pca.dropna(subset=['cycle time'], inplace=True)
                results_pca['lot'] = readout_file.loc[ro_ind, 'Lot Name']
                results_pca['wafer'] = readout_file.loc[ro_ind, 'Wafer Number']
                results_pca['IC X'] = readout_file.loc[ro_ind, 'IC X']
                results_pca['IC Y'] = readout_file.loc[ro_ind, 'IC Y']
                results_pca['Set temperature [C]'] = readout_file.loc[ro_ind, 'Set temperature [C]']
                results_pca['PS_VDD set voltage [mV]'] = readout_file.loc[ro_ind, 'PS_VDD set voltage [mV]']
                results_pca['Test Stage'] = readout_file.loc[ro_ind, 'Test Stage']
                results_pca['Sample Time'] = readout_file.loc[ro_ind, 'Sample Time']
                results_pca['Test Name'] = readout_file.loc[ro_ind, 'Test Name']
                results_pca['Soft Bin'] = readout_file.loc[ro_ind, 'Soft Bin']
                results_pca['Hard Bin'] = readout_file.loc[ro_ind, 'Hard Bin']
                results_pca['iddq meas'] = readout_file.loc[ro_ind, 'iddq meas']

                results_pca['cycle time'] = results_pca['cycle time'] + np.random.normal(0, 0.3, len(results_pca))

                results_pca_agg = results_pca.groupby(['lot', 'wafer', 'IC X', 'IC Y', 'Device', 'mode', 'Set temperature [C]', 'PS_VDD set voltage [mV]', 'Test Stage', 'Sample Time', 'Test Name', 'Soft Bin', 'Hard Bin', 'iddq meas'])[
                    'cycle time'].agg(['mean', 'std']).reset_index()

                concatenated_results_pca = pd.concat([concatenated_results_pca, results_pca_agg])

                rows_processed += 1
                progress_value = rows_processed / len(readout_file)
                progress_bar.progress(progress_value)
        st.write("Saving Processed Data...")
        # filename = f'https://raw.githubusercontent.com/dudilu/platform/main/{results_pca['lot'].iloc[0]}.csv'
        concatenated_results_pca.to_csv(filename, index=False)

        concatenated_results_pca['Device_mode'] = concatenated_results_pca['Device'] + '_' + concatenated_results_pca['mode']
        pivot_df = concatenated_results_pca.pivot_table(index=['lot', 'wafer', 'IC X', 'IC Y', 'Set temperature [C]', 'PS_VDD set voltage [mV]', 'Test Stage', 'Sample Time', 'Test Name', 'Soft Bin', 'Hard Bin', 'iddq meas'], columns='Device_mode', values='mean').reset_index()

        devices = ['nelvt', 'nlvt', 'nlvtll', 'nsvt', 'nulvt', 'nulvtll', 'pelvt', 'plvt', 'plvtll', 'psvt', 'pulvt', 'pulvtll']

        for device in devices:
            leak_col = f'{device}_leak'
            ref_col = f'{device}_ref'
            ratio_col = f'{device}_ratio'
            one_over_ratio_col = f'{device}_1/ratio'

            pivot_df[ratio_col] = pivot_df[ref_col] / pivot_df[leak_col]
            pivot_df[one_over_ratio_col] = pivot_df[leak_col] / pivot_df[ref_col]

        coeffs_file = "https://raw.githubusercontent.com/dudilu/platform/main/iddq_model_coeff.txt"
        coeffs = {}
        with open(coeffs_file, "r") as file:
            for line in file:
                key, value = line.strip().split(": ")
                coeffs[key.strip()] = float(value)

        pivot_df['iddq_est'] = 0
        for col, coeff in coeffs.items():
            pivot_df['iddq_est'] = pivot_df['iddq_est'] + pivot_df[col] * coeff

        # filename = f'C:\\Users\\DudiLubton\\PycharmProjects\\pythonProject\\platform\\agg1\\iddq\\{results_pca['lot'].iloc[0]}.csv'
        pivot_df.to_csv(filename, index=False)

        st.write("Processed Data Saved Successfully!")


