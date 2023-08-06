import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageColor
import colorsys
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans, BisectingKMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE


model_dict = {
    "KMeans": KMeans,
    "BisectingKMeans" : BisectingKMeans,
    "GaussianMixture": GaussianMixture,
    "MiniBatchKMeans": MiniBatchKMeans,
}

center_method = {
    "KMeans": "cluster_centers_",
    "BisectingKMeans" : "cluster_centers_",
    "GaussianMixture": "means_",
    "MiniBatchKMeans": "cluster_centers_",
}

n_cluster_arg = {
    "KMeans": "n_clusters",
    "BisectingKMeans" : "n_clusters",
    "GaussianMixture": "n_components",
    "MiniBatchKMeans": "n_clusters",

}

enhancement_range = {
    "Color": [0., 5., 0.2], 
    "Sharpness": [0., 3., 0.2], 
    "Contrast": [0.5, 1.5, 0.1], 
    "Brightness": [0.5, 1.5, 0.1]
}

sort_func_dict = {
    "rgb": (lambda r,g,b: (r, g, b)),
    "sum_rgb": (lambda r,g,b: r+g+b),
    "sqr_rgb": (lambda r,g,b: r**2+g**2+b**2),
    "hsv": (lambda r, g, b : colorsys.rgb_to_hsv(r, g, b)),
    "random": (lambda r, g, b: np.random.random())
}

def get_df_rgb(img, sample_size):
    """construct a sample RGB dataframe from image"""

    n_dims = np.array(img).shape[-1]
    r,g,b = np.array(img).reshape(-1,n_dims).T
    df = pd.DataFrame({"R": r, "G": g, "B": b}).sample(n=sample_size)
    return df

@st.cache_data
def get_palette(df_rgb, model_name, palette_size, sort_func="random"):
    """cluster pixels together and return a sorted color palette."""
    params = {n_cluster_arg[model_name]: palette_size}
    model = model_dict[model_name](**params)

    clusters = model.fit_predict(df_rgb)
        
    palette = getattr(model, center_method[model_name]).astype(int).tolist()
    
    palette.sort(key=lambda rgb : sort_func_dict[sort_func.rstrip("_r")](*rgb), 
                reverse=bool(sort_func.endswith("_r")))

    return palette

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def show_palette(palette_hex):
    """show palette strip"""
    palette = np.array([ImageColor.getcolor(color, "RGB") for color in  palette_hex])
    fig, ax = plt.subplots(dpi=100)
    ax.imshow(palette[np.newaxis, :, :])
    ax.axis('off')
    return fig


def store_palette(palette):
    """store palette colors in session state"""
    palette_size = len(palette)
    columns = st.columns(palette_size)
    for i, col in enumerate(columns):
        with col:        
            st.session_state[f"col_{i}"]= st.color_picker(label=str(i), value=rgb_to_hex(palette[i]), key=f"pal_{i}")

def display_matplotlib_code(palette_hex):

    st.write('Use this snippet in your code to make your color palette more sophisticated!')
    code = st.code(f"""
import matplotlib as mpl
from cycler import cycler

palette = {palette_hex}
mpl.rcParams["axes.prop_cycle"] = cycler(color=palette)
    """
    )   

def display_plotly_code(palette_hex):
    st.write('Use this snippet in your code to make your color palette more sophisticated!')
    st.code(f"""
import plotly.io as pio
import plotly.graph_objects as go
pio.templates["sophisticated"] = go.layout.Template(
    layout=go.Layout(
    colorway={palette_hex}
    )
)
pio.templates.default = 'sophisticated'
            """)

def plot_rgb_3d(df_rgb):
    """plot the sampled pixels in 3D RGB space"""

    if df_rgb.shape[0] > 2000:
        st.error("RGB plot can only be used for less than 2000 sample pixels.")
    else:
        colors = df_rgb.apply(rgb_to_hex, axis=1)
        fig = px.scatter_3d(df_rgb, x='R', y='G', z='B',
                color=colors, size=[1]*df_rgb.shape[0],
                opacity=0.7)

        st.plotly_chart(fig)


def plot_hsv_3d(df):
    """plot the sampled pixels in 3D RGB space"""
    df_rgb = df.copy()
    if df_rgb.shape[0] > 2000:
        st.error("RGB plot can only be used for less than 2000 sample pixels.")

    else:
        df_rgb[["H","S",'V']]= df_rgb.apply(lambda x: pd.Series(colorsys.rgb_to_hsv(x.R/255.,x.G/255.,x.B/255.)).T, axis=1)
        st.dataframe(df_rgb[["H","S",'V']])
        colors = df_rgb[["R","G","B"]].apply(rgb_to_hex, axis=1)
        fig = px.scatter_3d(df_rgb, x='H', y='S', z='V',
                color=colors, size=[1]*df_rgb.shape[0],
                opacity=0.7)

        st.plotly_chart(fig)

def print_praise():
    """Yes, I'm that vain and superficial! 🙄 """

    praise_quotes = [
        '"When I stumbled upon this app, it was like I found a *pearl* among the oysetrs. Absolutely stunning! "\n\n-- Johannes Merveer',
        '"I wish *Mona* was alive to see this masterpiece! I\'m sure she would have *smiled* at it..."\n\n-- Leonarda va Dinci',
        '"I\'m sorry, what was that? Ah yes, great app. I use it every *night*. Five *stars*!"\n\n-- Vincent van Vogue',
        '"We\'ve all been waiting years for an app to make a *big splash* like this, and now it\'s finally here!\n[Can you hand me that towel please?]"\n\n-- David Hockknee',
        '"It makes such a great *impression* on you, doesn\'t it? I know where I\'ll be getting my palette for painting the next *sunrise*!"\n\n-- Cloud Moanet',
        '"Maybe some other time... [Can I get a gin and tonic please?]"\n\n-- Edward Jumper',
    ]

    title = "[imaginary] **Praise for Sophisticated Palette**\n\n"
    # random_index = np.random.randint(len(praise_quotes))
    weights = np.array([2, 3.5, 3, 3, 3, 1])
    weights = weights/np.sum(weights)

    return title + np.random.choice(praise_quotes, p=weights)