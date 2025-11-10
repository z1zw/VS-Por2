import streamlit as st
import pandas as pd
import numpy as np
import chardet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from scipy.stats import linregress

st.set_page_config(page_title="BLWP Visual Analytics", layout="wide")

@st.cache_data
def load_df(file):
    raw = file.read()
    enc = chardet.detect(raw)['encoding']
    file.seek(0)
    df = pd.read_csv(file, encoding=enc)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r'[\s\-]+', '_', regex=True)
        .str.replace('\ufeff', '')
    )
    date_col_candidates = [c for c in df.columns if 'date' in c]
    if not date_col_candidates:
        return df
    df.rename(columns={date_col_candidates[0]: 'sample_date'}, inplace=True)
    df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
    loc_candidates = [c for c in df.columns if 'loc' in c or 'site' in c or 'station' in c]
    if not loc_candidates:
        return df
    df.rename(columns={loc_candidates[0]: 'location'}, inplace=True)
    meas_candidates = [c for c in df.columns if 'measure' in c or 'chemical' in c or 'substance' in c]
    if not meas_candidates:
        return df
    df.rename(columns={meas_candidates[0]: 'measure'}, inplace=True)
    val_candidates = [c for c in df.columns if 'value' in c or 'reading' in c or 'concentration' in c]
    if not val_candidates:
        return df
    df.rename(columns={val_candidates[0]: 'value'}, inplace=True)
    df['year'] = df['sample_date'].dt.year
    df['month'] = df['sample_date'].dt.month
    df['ym'] = df['sample_date'].dt.to_period('M').astype(str)
    return df

def robust_z(s):
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med)) or 1e-9
    return (s - med) / (1.4826 * mad)

st.sidebar.header("Data")
readings_file = st.sidebar.file_uploader("Upload readings CSV", type=["csv"])
units_file = st.sidebar.file_uploader("Upload units CSV", type=["csv"])

df = None
if readings_file is not None:
    df = load_df(readings_file)
    if "location" in df.columns:
        if units_file is not None:
            units_raw = units_file.read()
            enc2 = chardet.detect(units_raw)['encoding']
            units_file.seek(0)
            units = pd.read_csv(units_file, encoding=enc2)
            units.columns = [c.strip().lower().replace(' ', '_') for c in units.columns]
            if "measure" in units.columns:
                df = df.merge(units, on="measure", how="left")
        st.session_state["df"] = df

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([ "Data Normalization", "Trend Analysis", "Evolution Explorer", "Comparative Regimes", "Risk Graph", "Attention Maps", "Chemical Graph Evolution" ])
with tab1:
    st.subheader("Data Normalization & Diagnostics")
    if "df" in st.session_state:
        df = st.session_state["df"].copy()
        norm = st.selectbox("Normalization method", ["Robust Z (per measure)", "Quantile 0–1 (per measure)"])
        if norm.startswith("Robust"):
            df['value_norm'] = df.groupby('measure')['value'].transform(robust_z)
        else:
            df['value_norm'] = df.groupby('measure')['value'].transform(
                lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
            )
        st.session_state["df_norm"] = df
        st.success("Normalization applied and cached successfully")
        st.dataframe(df.head(20))
    else:
        st.warning("Please upload data first.")

with tab2:
    st.subheader("Auto-Trend Dashboard")

    if "df_norm" not in st.session_state:
        st.warning("Please normalize data first in 'Data & Units' tab.")
    else:
        d = st.session_state["df_norm"].copy()

        agg = (
            d.groupby(['location', 'year'], as_index=False)['value_norm']
             .sum()
             .rename(columns={'value_norm': 'year_sum'})
        )

        def get_trend(df):
            if len(df) < 2:
                return 0
            slope, _, _, _, _ = linregress(df['year'], df['year_sum'])
            return slope

        trend = agg.groupby('location').apply(get_trend).reset_index(name='trend')

        q1, q2 = trend['trend'].quantile([0.33, 0.66])
        def auto_group(x):
            if x <= q1:
                return "Group 1 (Decreasing)"
            elif x <= q2:
                return "Group 2 (Stable)"
            else:
                return "Group 3 (Increasing)"
        trend['Group'] = trend['trend'].apply(auto_group)
        d = d.merge(trend[['location', 'Group']], on='location', how='left')

        pivot_sum = agg.pivot(index='location', columns='year', values='year_sum').fillna(0.0)
        mean_by_group_year = (
            d.groupby(['Group', 'year'], as_index=False)['value_norm']
             .mean()
             .rename(columns={'value_norm': 'mean_value'})
        )

        palette = {
            "Group 1 (Decreasing)": "#4575b4",
            "Group 2 (Stable)": "#ffffbf",
            "Group 3 (Increasing)": "#d73027"
        }

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "(a) PCA projection by trend groups",
                "(b) Annual Chemical Sum Matrix",
                "(c) Variable Loadings (PCA)",
                "(d) Group-wise Mean Time Trend",
                "(e) Mean Heatmap per Group-Year",
                "(f) Difference Map (Inc - Dec)"
            ],
            specs=[
                [{"type": "xy"}, {"type": "heatmap"}],
                [{"type": "heatmap"}, {"type": "xy"}],
                [{"type": "heatmap"}, {"type": "heatmap"}],
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )

        X = d.pivot_table(index=['sample_date', 'location'], columns='measure', values='value_norm').fillna(0.0)
        if len(X) >= 2:
            pca = PCA(n_components=2).fit(X.values)
            proj = pca.transform(X.values)
            proj_df = pd.DataFrame(proj, columns=['PC1', 'PC2'], index=X.index).reset_index()
            proj_df = proj_df.merge(trend[['location', 'Group']], on='location', how='left')
            for g, sub in proj_df.groupby('Group'):
                fig.add_trace(
                    go.Scatter(
                        x=sub['PC1'], y=sub['PC2'],
                        mode='markers',
                        marker=dict(size=5, color=palette.get(g, "#ccc"), opacity=0.7),
                        name=g
                    ), row=1, col=1
                )

        fig.add_trace(
            go.Heatmap(
                z=pivot_sum.values,
                x=pivot_sum.columns,
                y=pivot_sum.index,
                colorscale='RdYlBu_r',
                colorbar=dict(title="Sum", len=0.35)
            ), row=1, col=2
        )

        if len(X.columns) > 1:
            loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=X.columns)
            fig.add_trace(
                go.Heatmap(
                    z=loadings.values,
                    x=loadings.columns,
                    y=loadings.index,
                    colorscale='RdBu',
                    colorbar=dict(title="Loading", len=0.35)
                ), row=2, col=1
            )

        for g, sub in mean_by_group_year.groupby('Group'):
            fig.add_trace(
                go.Scatter(
                    x=sub['year'], y=sub['mean_value'],
                    mode='lines+markers',
                    name=g,
                    line=dict(width=3, color=palette.get(g))
                ), row=2, col=2
            )

        heat_data = (
            d.groupby(['Group', 'year', 'measure'], as_index=False)['value_norm']
             .mean()
             .pivot_table(index=['Group', 'year'], columns='measure', values='value_norm')
             .fillna(0.0)
        )
        z = heat_data.values
        y = [f"{g}-{y}" for g, y in heat_data.index]
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=heat_data.columns,
                y=y,
                colorscale='RdBu_r',
                colorbar=dict(title="Mean Value", len=0.35)
            ), row=3, col=1
        )

        inc = heat_data.loc[heat_data.index.get_level_values(0) == 'Group 3 (Increasing)'].mean()
        dec = heat_data.loc[heat_data.index.get_level_values(0) == 'Group 1 (Decreasing)'].mean()
        diff = (inc - dec).to_frame(name='Diff').T
        fig.add_trace(
            go.Heatmap(
                z=diff.values,
                x=diff.columns,
                y=['Δ(Inc–Dec'],
                colorscale='RdBu',
                colorbar=dict(title="Δ", len=0.35)
            ), row=3, col=2
        )

        fig.update_layout(
            height=1350,
            paper_bgcolor="#101417",
            plot_bgcolor="#181d21",
            font=dict(size=12, color="#e5e5e5", family="Arial"),
            title=dict(
                text="TULCA Auto-Trend Multi-Panel Dashboard (Location × Year Chemical Dynamics)",
                x=0.45,
                font=dict(size=18, color="#f5f5f5")
            ),
            margin=dict(l=60, r=60, t=100, b=70),
            hovermode="closest",
            template="plotly_dark",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="#555",
                borderwidth=1,
                font=dict(size=11, color="#e5e5e5")
            ),
            showlegend=True
        )

        for axis in fig.layout:
            if axis.startswith("xaxis") or axis.startswith("yaxis"):
                fig.layout[axis].showgrid = True
                fig.layout[axis].gridcolor = "#3a3f46"
                fig.layout[axis].zeroline = False
                fig.layout[axis].linecolor = "#555b62"
                fig.layout[axis].tickcolor = "#555b62"
                fig.layout[axis].mirror = True

        for trace in fig.data:
            if hasattr(trace, "type") and trace.type == "scatter":
                if hasattr(trace, "mode") and "lines+markers" in trace.mode:
                    trace.showlegend = True
                else:
                    trace.showlegend = False
            else:
                trace.showlegend = False

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Summary")
        st.write(f"- Total locations: {d['location'].nunique()}")
        st.write(f"- Groups formed: {trend['Group'].value_counts().to_dict()}")
        st.write(f"- Trend range: {trend['trend'].min():.3f} → {trend['trend'].max():.3f}")
        st.divider()
        st.markdown("### Detailed Trend Group Results")

        trend_sorted = (
            trend.sort_values("trend", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            trend_sorted.style.background_gradient(subset=["trend"], cmap="RdBu_r"),
            use_container_width=True,
            height=400
        )

        st.markdown("### Group-wise Summary")
        group_summary = (
            agg.merge(trend[["location", "Group"]], on="location", how="left")
            .groupby("Group")["year_sum"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        numeric_cols = group_summary.select_dtypes(include=[np.number]).columns
        st.dataframe(
            group_summary.style.format(subset=numeric_cols, formatter="{:.3f}"),
            use_container_width=True
        )

        st.markdown("### 🗺Locations by Group")
        for g in trend["Group"].unique():
            subset = trend_sorted[trend_sorted["Group"] == g]
            loc_list = ", ".join(subset["location"].tolist())
            st.markdown(f"**{g}** ({len(subset)} locations):")
            st.write(loc_list)
            st.write("---")

        csv_export = trend_sorted.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Trend Group Table (CSV)",
            data=csv_export,
            file_name="trend_group_results.csv",
            mime="text/csv"
        )
with tab3:
    st.subheader("Evolution Dashboard")
    if "df_norm" not in st.session_state:
        st.warning("Please normalize data first in 'Data & Units' tab.")
    else:
        d = st.session_state["df_norm"].copy()
        st.caption("Explore chemical evolution, cross-correlation.")

        locs = sorted(d["location"].unique().tolist())
        chems = sorted(d["measure"].unique().tolist())
        col1, col2 = st.columns(2)
        sel_loc = col1.selectbox("Select Location", ["ALL"] + locs)
        sel_chem = col2.multiselect("Select Chemicals", chems[:8], default=chems[:3])

        if not sel_chem:
            st.warning("Please select at least one chemical.")
            st.stop()

        if sel_loc != "ALL":
            d = d[d["location"] == sel_loc]
        agg = (
            d.groupby(["year", "measure"], as_index=False)["value_norm"]
             .agg(["mean", "std"])
             .reset_index()
        )

        fig1 = go.Figure()
        for m in sel_chem:
            sub = agg[agg["measure"] == m]
            fig1.add_trace(go.Scatter(
                x=sub["year"], y=sub["mean"],
                mode="lines+markers", name=f"{m} mean",
                line=dict(width=2)
            ))
            fig1.add_trace(go.Scatter(
                x=sub["year"].tolist() + sub["year"].tolist()[::-1],
                y=(sub["mean"]+sub["std"]).tolist() + (sub["mean"]-sub["std"]).tolist()[::-1],
                fill="toself", fillcolor="rgba(44,160,44,0.2)",
                line=dict(width=0), name=f"{m} ±1σ", showlegend=False
            ))
        fig1.update_layout(
            title="(a) Temporal Evolution per Chemical (Mean ± 1σ)",
            xaxis_title="Year", yaxis_title="Normalized Value (Z)",
            template="plotly_dark", height=400,
            legend=dict(orientation="h", y=-0.25)
        )

        corr_data = (
            d.groupby(["year", "measure"], as_index=False)["value_norm"]
             .mean().pivot(index="year", columns="measure", values="value_norm")
        )
        corr_mat = corr_data.corr().fillna(0)
        fig2 = go.Figure(
            data=go.Heatmap(
                z=corr_mat.values, x=corr_mat.columns, y=corr_mat.index,
                colorscale="RdBu", zmid=0, colorbar=dict(title="r")
            )
        )
        fig2.update_layout(
            title="(b) Cross-Correlation Between Chemicals (Annual Mean)",
            template="plotly_dark", height=400,
            margin=dict(l=60, r=60, t=60, b=60)
        )

        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        loc_avg = d.groupby("year")["value_norm"].mean().reset_index()
        fig3.add_trace(go.Bar(
            x=loc_avg["year"], y=loc_avg["value_norm"],
            name="Overall Mean (All Chemicals)",
            marker_color="rgba(55,126,184,0.6)"
        ), secondary_y=False)
        chem_sel = d[d["measure"] == sel_chem[0]].groupby("year")["value_norm"].mean().reset_index()
        fig3.add_trace(go.Scatter(
            x=chem_sel["year"], y=chem_sel["value_norm"],
            mode="lines+markers", line=dict(color="orange", width=3),
            name=f"{sel_chem[0]} (Selected)"
        ), secondary_y=True)
        fig3.update_layout(
            title=f"(c) Dual-Axis Comparison ({sel_chem[0]} vs All-Chemical Mean)",
            template="plotly_dark", height=400,
            legend=dict(orientation="h", y=-0.25)
        )
        fig3.update_yaxes(title_text="Overall Mean (Z)", secondary_y=False)
        fig3.update_yaxes(title_text=f"{sel_chem[0]} Value (Z)", secondary_y=True)

        yearly_var = (
            d.groupby(["year", "measure"], as_index=False)["value_norm"]
             .mean().pivot(index="year", columns="measure", values="value_norm")
        ).var(axis=1)
        rolling_var = yearly_var.rolling(window=3, center=True).mean()

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=yearly_var.index, y=yearly_var.values,
            name="Annual Chemical Variance", marker_color="#1f78b4"
        ))
        fig4.add_trace(go.Scatter(
            x=rolling_var.index, y=rolling_var.values,
            mode="lines", line=dict(color="#ff7f00", width=3),
            name="3-Year Smoothed Variance"
        ))
        fig4.update_layout(
            title="(d) Yearly Chemical Variance Spectrum (Chemical Diversity Energy)",
            xaxis_title="Year", yaxis_title="Variance (Z²)",
            template="plotly_dark", height=400,
            legend=dict(orientation="h", y=-0.25)
        )

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
with tab4:
    st.subheader("Comparative Regimes")
    st.caption("Compare temporal regimes (e.g., early vs late years) via contrastive PCA and trajectory analysis.")

    if "df_norm" not in st.session_state:
        st.warning("Please normalize data first.")
    else:
        import sklearn.decomposition as skd
        d = st.session_state["df_norm"].copy()
        years = sorted(d["year"].dropna().unique().tolist())

        st.markdown("### Step 1. Regime Definition")
        colA, colB, colC = st.columns(3)
        regime_A = colA.multiselect("Regime A years", years[:max(1, len(years)//3)], default=years[:min(3, len(years)//3)])
        regime_B = colB.multiselect("Regime B years", years[max(1, len(years)//3):2*max(1, len(years)//3)],
                                    default=years[len(years)//3:2*len(years)//3])
        regime_C = colC.multiselect("Regime C years", years[2*max(1, len(years)//3):], default=years[-min(3, len(years)//3):])

        st.markdown("**Regime Summary:**")
        regimes_info = pd.DataFrame({
            "Regime": ["A", "B", "C"],
            "Years": [", ".join(map(str, regime_A)), ", ".join(map(str, regime_B)), ", ".join(map(str, regime_C))],
            "n_years": [len(regime_A), len(regime_B), len(regime_C)]
        })
        st.dataframe(regimes_info, use_container_width=True, hide_index=True)
        st.markdown("### 📅 Regime Year Ranges")
        st.write(f"**Regime A ({len(regime_A)} years)**: {', '.join(map(str, regime_A)) if regime_A else 'None'}")
        st.write(f"**Regime B ({len(regime_B)} years)**: {', '.join(map(str, regime_B)) if regime_B else 'None'}")
        st.write(f"**Regime C ({len(regime_C)} years)**: {', '.join(map(str, regime_C)) if regime_C else 'None'}")
        st.divider()

        d["col"] = d["measure"] + " @ " + d["location"]
        X = d.pivot_table(index="sample_date", columns="col", values="value_norm", aggfunc="mean").sort_index().fillna(0)
        ylab = X.index.year.map(
            lambda y: 0 if y in regime_A else (1 if y in regime_B else (2 if y in regime_C else np.nan))
        )
        mask = ~np.isnan(ylab)
        X2, y = X[mask], ylab[mask].astype(int)

        if len(np.unique(y)) < 2:
            st.warning("Please select at least two regimes with samples.")
        else:
            pca_list = [skd.PCA(n_components=2).fit(X2[y == k]) for k in np.unique(y)]
            delta_AB = pca_list[1].components_ - pca_list[0].components_
            proj_AB = X2 @ delta_AB.T
            load_AB = pd.Series(delta_AB[0], index=X2.columns)

            delta_BC = None
            if len(pca_list) == 3:
                delta_BC = pca_list[2].components_ - pca_list[1].components_

            pos_load = load_AB.nlargest(20)
            neg_load = load_AB.nsmallest(20)

            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    "(a) Contrastive Projection (Δ between Regime B–A)",
                    "(b) Variable × Component (Δ Loadings)",
                    "(c) Instance × Component Map",
                    "(d) Top + Variables (B>A)",
                    "(e) Top − Variables (A>B)",
                    "(f) Mean Trajectories by Regime"
                ],
                specs=[
                    [{"type": "xy"}, {"type": "heatmap"}],
                    [{"type": "heatmap"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "xy"}]
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.12
            )

            colors = ["#2b8cbe", "#de2d26", "#41ab5d"]
            fig.add_trace(go.Scatter(
                x=proj_AB.iloc[:, 0], y=proj_AB.iloc[:, 1],
                mode="markers",
                marker=dict(size=6, color=[colors[i] for i in y], opacity=0.8),
                name="Contrastive Projection"
            ), row=1, col=1)

            for i, name in enumerate(["Regime A", "Regime B", "Regime C"][:len(np.unique(y))]):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=8, color=colors[i]), name=name
                ), row=1, col=1)

            heat_var = np.vstack([delta_AB[0][:50], delta_AB[1][:50]])
            fig.add_trace(go.Heatmap(
                z=heat_var, y=["C1", "C2"], x=X2.columns[:50],
                colorscale="RdBu", zmid=0, colorbar=dict(title="Δ Loading", len=0.3)
            ), row=1, col=2)

            fig.add_trace(go.Heatmap(
                z=proj_AB.values.T,
                x=proj_AB.index.astype(str),
                y=["Comp1", "Comp2"],
                colorscale="RdBu", zmid=0,
                colorbar=dict(title="Projection", len=0.3)
            ), row=2, col=1)

            fig.add_trace(go.Bar(x=pos_load.index, y=pos_load.values, marker_color="#fc8d62", name="B>A"), row=2, col=2)
            fig.add_trace(go.Bar(x=neg_load.index, y=neg_load.values, marker_color="#66c2a5", name="A>B"), row=3, col=1)

            d_mean = d.groupby(["year", "measure"])["value_norm"].mean().reset_index()
            for reg, clr, label in zip([regime_A, regime_B, regime_C], colors, ["Regime A", "Regime B", "Regime C"]):
                if len(reg) > 0:
                    sub = d_mean[d_mean["year"].isin(reg)]
                    fig.add_trace(go.Scatter(
                        x=sub["year"], y=sub["value_norm"],
                        mode="lines+markers", line=dict(color=clr, width=2),
                        name=label
                    ), row=3, col=2)

            fig.update_layout(
                height=1500,
                template="plotly_dark",
                font=dict(size=12, color="#e5e5e5"),
                margin=dict(l=70, r=40, t=100, b=80),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)
                ),
                title=dict(
                    text="Multi-Regime Contrastive Analysis (A–B–C)",
                    x=0.45,
                    font=dict(size=20, color="#f5f5f5")
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Regime-wise Statistical Summary")
            df_stats = d.groupby("year")["value_norm"].mean().reset_index()
            df_stats["Regime"] = df_stats["year"].apply(
                lambda y: "A" if y in regime_A else ("B" if y in regime_B else ("C" if y in regime_C else "N/A"))
            )
            regime_summary = df_stats.groupby("Regime")["value_norm"].agg(["mean", "std", "min", "max"]).reset_index()
            numeric_cols = regime_summary.select_dtypes(include=[np.number]).columns
            st.dataframe(
                regime_summary.style.format({col: "{:.3f}" for col in numeric_cols}),
                use_container_width=True
            )

            st.markdown("### Top Differential Variables (Δ between B and A)")
            st.dataframe(pd.concat([pos_load, neg_load], axis=1, keys=["Positive (B>A)", "Negative (A>B)"]),
                         use_container_width=True)

with tab5:
    st.subheader("Risk Propagation Graph • Temporal Diffusion + Statistics")

    if "df_norm" not in st.session_state:
        st.warning("Please upload and normalize data first.")
    else:
        d = st.session_state["df_norm"].copy()
        measures = sorted(d["measure"].dropna().unique().tolist())
        measure_sel = st.selectbox("Select chemical measure", measures, index=0)
        lag_max = st.slider("Maximum lag (months)", 1, 12, 3)
        corr_thresh = st.slider("Correlation threshold", 0.2, 0.9, 0.6, 0.05)
        win = st.slider("Rolling window (months)", 1, 12, 3)

        d1 = d[d["measure"] == measure_sel].dropna(subset=["sample_date", "value_norm"])
        X = d1.pivot_table(index="sample_date", columns="location", values="value_norm", aggfunc="mean")
        X = X.sort_index().resample("M").mean().rolling(win, min_periods=1).mean().interpolate(limit_direction="both")

        import itertools, networkx as nx
        edges = []
        for a, b in itertools.permutations(X.columns, 2):
            best, lag_best = 0, 0
            for lag in range(1, lag_max+1):
                c = X[a].corr(X[b].shift(lag))
                if pd.notna(c) and abs(c) > abs(best):
                    best, lag_best = c, lag
            if best > corr_thresh:
                edges.append((a, b, best, lag_best))

        G = nx.DiGraph()
        for (a, b, w, l) in edges:
            G.add_edge(a, b, weight=w, lag=l)

        if len(G.nodes) == 0:
            st.warning("No significant propagation edges found.")
        else:
            pos = nx.spring_layout(G, k=0.7, seed=7)
            node_names = list(G.nodes())
            edge_x, edge_y, edge_w, edge_lag = [], [], [], []
            for (u, v, d2) in G.edges(data=True):
                x0, y0 = pos[u]; x1, y1 = pos[v]
                edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
                edge_w.append(d2["weight"]); edge_lag.append(d2["lag"])

            months = X.index.to_period("M").astype(str).tolist()
            zscores = ((X - X.median())/(1.4826*(X.sub(X.median()).abs().median()+1e-9))).clip(-4, 4)
            sizes, colors = [], []
            for t in X.index:
                row = zscores.loc[t].reindex(node_names).fillna(0)
                sizes.append([14 + 5*float(max(v,0)) for v in row.values])
                colors.append([float(v) for v in row.values])

            frames = []
            for i, m in enumerate(months):
                frames.append(go.Frame(
                    data=[
                        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.8, color="#888")),
                        go.Scatter(x=[pos[n][0] for n in node_names], y=[pos[n][1] for n in node_names],
                                   mode="markers+text", text=node_names, textposition="top center",
                                   marker=dict(size=sizes[i], color=colors[i],
                                               colorscale="RdBu", cmin=-3, cmax=3, line=dict(width=1,color="white")))
                    ],
                    name=m
                ))

            fig = make_subplots(rows=2, cols=2, specs=[[{"type":"xy"}, {"type":"bar"}],
                                                      [{"type":"scatter"}, {"type":"scatter"}]],
                                subplot_titles=["Diffusion Network (animation)",
                                                "Edge Weight Distribution",
                                                "Lag vs Strength Scatter",
                                                "Degree Centrality Map"])

            base_edges = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.8, color="#888"))
            base_nodes = go.Scatter(x=[pos[n][0] for n in node_names], y=[pos[n][1] for n in node_names],
                                    mode="markers+text", text=node_names, textposition="top center",
                                    marker=dict(size=sizes[0], color=colors[0],
                                                colorscale="RdBu", cmin=-3, cmax=3, line=dict(width=1,color="white")))
            fig.add_trace(base_edges, row=1, col=1)
            fig.add_trace(base_nodes, row=1, col=1)

            fig.add_trace(go.Histogram(x=edge_w, nbinsx=20, marker_color="#66c2a5"), row=1, col=2)
            fig.add_trace(go.Scatter(x=edge_lag, y=edge_w, mode="markers", marker=dict(size=8,color="#fc8d62",opacity=0.7)), row=2, col=1)
            cent = nx.degree_centrality(G)
            fig.add_trace(go.Bar(x=list(cent.keys()), y=list(cent.values()), marker_color="#8da0cb"), row=2, col=2)

            fig.update(frames=frames)
            fig.update_layout(
                template="plotly_dark", height=900,
                title=f"Risk Propagation & Temporal Diffusion — {measure_sel}",
                showlegend=False,
                updatemenus=[dict(type="buttons", showactive=False, buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 600, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate", "transition": {"duration": 0}}])
                ])]
            )
            st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("RiverChem Transformer Analyzer")

    if "df_norm" not in st.session_state:
        st.warning("Please normalize data first in 'Data & Units' tab.")
    else:
        import torch
        d = st.session_state["df_norm"].copy()

        loc = st.selectbox("Select location", sorted(d["location"].unique()))
        chems = st.multiselect(
            "Select chemicals (multi-select for co-evolution map)",
            sorted(d["measure"].unique()),
            default=sorted(d["measure"].unique())[:4]
        )
        if not chems:
            st.stop()

        sub = d[(d["location"] == loc) & (d["measure"].isin(chems))]
        X = sub.pivot_table(index="ym", columns="measure", values="value_norm").fillna(0)
        if X.empty:
            st.warning("No data available for the selected location/chemicals.")
            st.stop()

        X_t = torch.tensor(X.values, dtype=torch.float32)
        Q, K, V = X_t, X_t, X_t

        attn_time = torch.softmax((Q @ K.T) / np.sqrt(X_t.shape[1]), dim=-1).detach().numpy()
        attn_chem = torch.softmax((X_t.T @ X_t) / np.sqrt(X_t.shape[0]), dim=-1).detach().numpy()

        hovertext = []
        for i, t1 in enumerate(X.index):
            row = []
            for j, t2 in enumerate(X.index):
                row.append(f"T1={t1}<br>T2={t2}<br>Attn={attn_time[i,j]:.3f}")
            hovertext.append(row)

        fig1 = go.Figure(data=go.Heatmap(
            z=attn_time,
            x=X.index, y=X.index,
            text=hovertext,
            hoverinfo="text",
            colorscale="Viridis",
            colorbar=dict(title="Temporal Attention", len=0.8)
        ))
        fig1.update_layout(
            title=f"Temporal Attention Map ({loc})",
            height=500,
            font=dict(size=12, color="#e5e5e5"),
            template="plotly_dark",
            margin=dict(l=70, r=60, t=80, b=50)
        )

        hoverchem = []
        for i, c1 in enumerate(X.columns):
            row = []
            for j, c2 in enumerate(X.columns):
                row.append(f"{c1} → {c2}<br>Attn={attn_chem[i,j]:.3f}")
            hoverchem.append(row)

        fig2 = go.Figure(data=go.Heatmap(
            z=attn_chem,
            x=X.columns, y=X.columns,
            text=hoverchem,
            hoverinfo="text",
            colorscale="RdBu",
            zmid=0.5,
            colorbar=dict(title="Chemical Attention", len=0.8)
        ))
        fig2.update_layout(
            title=f"Chemical Interaction Attention ({loc})",
            height=500,
            font=dict(size=12, color="#e5e5e5"),
            template="plotly_dark",
            margin=dict(l=70, r=60, t=80, b=50)
        )

        avg_attn = pd.Series(attn_chem.mean(axis=1), index=X.columns).sort_values(ascending=False)
        fig3 = go.Figure(go.Bar(
            x=avg_attn.index,
            y=avg_attn.values,
            marker_color="#66c2a5",
            text=[f"{v:.3f}" for v in avg_attn.values],
            textposition="auto"
        ))
        fig3.update_layout(
            title="Average Influence of Each Chemical",
            height=400,
            font=dict(size=12, color="#e5e5e5"),
            template="plotly_dark",
            margin=dict(l=60, r=40, t=70, b=70),
            xaxis_title="Chemical",
            yaxis_title="Mean Attention"
        )

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)

with tab7:
    st.subheader("Chemical Evolution Graph Explorer ")
    st.caption("Analyze chemical co-evolution and cross-location propagation patterns over time.")

    if "df_norm" not in st.session_state:
        st.warning("Please normalize data first in 'Data & Units' tab.")
    else:
        import networkx as nx
        import plotly.graph_objects as go

        d = st.session_state["df_norm"].copy()
        locs = ["ALL"] + sorted(d["location"].unique().tolist())
        loc_sel = st.multiselect("Select locations", locs, default=["ALL"])
        chems_sel = st.multiselect(
            "Select chemicals", sorted(d["measure"].unique()), default=sorted(d["measure"].unique())[:8]
        )
        agg_mode = st.radio("Aggregation Mode", ["Monthly", "Yearly"], horizontal=True)

        if "ALL" in loc_sel:
            sub = d[d["measure"].isin(chems_sel)]
        else:
            sub = d[d["location"].isin(loc_sel) & d["measure"].isin(chems_sel)]

        if agg_mode == "Yearly":
            sub["time_unit"] = sub["year"]
        else:
            sub["time_unit"] = sub["ym"]

        pivot = sub.pivot_table(index="time_unit", columns="measure", values="value_norm", aggfunc="mean").fillna(0)
        delta = pivot.diff().dropna()

        if len(delta) < 2:
            st.warning("Not enough temporal data for evolution analysis.")
            st.stop()

        corr = delta.corr().fillna(0)
        G = nx.Graph()

        for chem in corr.columns:
            G.add_node(chem, avg_change=delta[chem].mean())

        for i in corr.columns:
            for j in corr.columns:
                if i != j and abs(corr.loc[i, j]) > 0.5:
                    G.add_edge(i, j, weight=corr.loc[i, j])

        pos = nx.spring_layout(G, weight="weight", k=0.8, seed=42)

        edge_x, edge_y, edge_color = [], [], []
        for u, v, d_edge in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_color.append(d_edge["weight"])

        node_x, node_y, node_text, node_color = [], [], [], []
        for node, attr in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Δmean={attr['avg_change']:.3f}")
            node_color.append(attr["avg_change"])

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=1, color="#888"),
            hoverinfo="none"
        ))
        fig1.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(
                size=18,
                color=node_color,
                colorscale="RdBu",
                cmin=-max(abs(np.array(node_color))), cmax=max(abs(np.array(node_color))),
                line=dict(width=2, color="white")
            ),
            text=[n for n in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text"
        ))
        fig1.update_layout(
            title="Chemical Co-Evolution Graph (edges = |corr(ΔCi, ΔCj)| > 0.5)",
            height=650,
            showlegend=False,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=90, b=50)
        )
        st.plotly_chart(fig1, use_container_width=True)

        sel = st.multiselect("Select chemicals to view temporal evolution", chems_sel[:3])
        if sel:
            fig2 = go.Figure()
            for c in sel:
                fig2.add_trace(go.Scatter(
                    x=pivot.index.astype(str),
                    y=pivot[c],
                    mode="lines+markers",
                    name=c
                ))
            fig2.update_layout(
                title="Temporal Evolution of Selected Chemicals",
                height=450,
                template="plotly_dark",
                xaxis_title="Time",
                yaxis_title="Normalized Concentration"
            )
            st.plotly_chart(fig2, use_container_width=True)
