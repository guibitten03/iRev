from app import *

layout = html.Div([
            dcc.Store("ex_setup_tables"),
            html.Div(id = "default-input-callback_t"),
            
            dbc.Row([
                dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H3("Datasets")
                                        ]),
                                        
                                        dbc.Col([
                                            dbc.Button("Amazon", color = "primary", className = "me-1 col-6", id = "amz-btn-t")
                                            
                                        ], align = "center"),
                                        dbc.Col([
                                            dbc.Button("Yelp", color = "primary", className = "me-1 col-6", id = "yel-btn-t")

                                        ], align = "center")
                                        
                                    ]),
                                    
                                ])
                            ], className = "p-3"),
                            
                        ]),
                        
            ], className = "mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id = "table-ex", className = "p-5")
                            
                        ])
                    ])
                ])
            ])
])

@app.callback(
    Output("ex_setup_tables", "data"),
    Input("default-input-callback_t", "children")
)
def charge_dataframe(none_input):
    return pd.read_csv("Experimental_setup.csv").to_dict()


@app.callback(
    Output("table-ex", "children"),
    
    [
        Input("amz-btn-t", "n_clicks"),
        Input("yel-btn-t", "n_clicks"),
        Input("ex_setup_tables", "data"),
    ],
    
)
def charge_table(amz, yel, data):
    df = pd.DataFrame(data)
    
    dataset = "Amazon_Music" if ctx.triggered_id == "amz-btn-t" else "Yelp" 
    
    df = df.loc[df['Dataset'] == dataset]
    
    return dash.dash_table.DataTable(
        df.to_dict('records'),
        [
            {"name": i, "id": i} for i in df.columns
        ],
        style_as_list_view=True,
        style_header={
        'backgroundColor': 'rgb(32,32,32)',
        'color': 'white'
        },
        style_data={
            'backgroundColor': 'rgb(0,0,0)',
            'color': 'white'
        },
    )
    
    