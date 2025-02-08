import React from 'react';
import { Paper, Grid, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

function VisualizationDashboard({ data, colorScheme }) {
    const isDark = colorScheme === 'dark';
    const textColor = isDark ? '#C1C2C5' : '#1A1B1E';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.1)';
    const axisColor = isDark ? '#5C5F66' : '#868E96';

    // Define color schemes for different chart types
    const chartColors = {
        primary: isDark ? '#4DABF7' : '#228BE6',      // Blue
        secondary: isDark ? '#FF8787' : '#FA5252',    // Red
        accent: isDark ? '#FFD43B' : '#FCC419',       // Yellow
        success: isDark ? '#69DB7C' : '#40C057',      // Green
        neutral: isDark ? '#909296' : '#495057',      // Gray
    };

    const renderPlot = (plotData, metadata) => {
        if (!plotData || !metadata) return null;

        // Enhance plot data with better colors
        const enhancedPlotData = plotData.map((trace, index) => ({
            ...trace,
            marker: {
                ...trace.marker,
                color: trace.marker?.color || chartColors[Object.keys(chartColors)[index % Object.keys(chartColors).length]]
            },
            line: {
                ...trace.line,
                color: trace.line?.color || chartColors[Object.keys(chartColors)[index % Object.keys(chartColors).length]]
            }
        }));

        const layout = {
            title: {
                text: metadata.title || '',
                font: {
                    color: textColor,
                    size: 18
                },
                y: 0.9
            },
            showlegend: true,
            legend: {
                font: { color: textColor },
                bgcolor: 'transparent'
            },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: { t: 60, r: 30, b: 50, l: 60 },
            xaxis: {
                title: metadata.x_label || '',
                showgrid: true,
                gridcolor: gridColor,
                zeroline: true,
                zerolinecolor: axisColor,
                zerolinewidth: 1,
                tickfont: { color: textColor },
                titlefont: { color: textColor },
                linecolor: axisColor,
                tickcolor: axisColor
            },
            yaxis: {
                title: metadata.y_label || '',
                showgrid: true,
                gridcolor: gridColor,
                zeroline: true,
                zerolinecolor: axisColor,
                zerolinewidth: 1,
                tickfont: { color: textColor },
                titlefont: { color: textColor },
                linecolor: axisColor,
                tickcolor: axisColor
            },
            font: { color: textColor },
            modebar: {
                bgcolor: 'transparent',
                color: textColor,
                activecolor: chartColors.primary
            }
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: [
                'lasso2d',
                'select2d',
                'hoverClosestCartesian',
                'hoverCompareCartesian',
                'toggleSpikelines'
            ],
            toImageButtonOptions: {
                format: 'png',
                filename: metadata.title || 'plot',
                height: 800,
                width: 1200,
                scale: 2
            },
            displaylogo: false
        };

        return (
            <Plot
                data={enhancedPlotData}
                layout={layout}
                config={config}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
            />
        );
    };

    if (!data || !data.visualizations || !data.visualization_metadata) {
        return (
            <Text c={isDark ? 'dimmed' : 'dimmed'} align="center">
                No visualization data available
            </Text>
        );
    }

    return (
        <Grid>
            {Object.entries(data.visualizations).map(([key, plotData], index) => (
                <Grid.Col key={key} span={12}>
                    <Paper
                        shadow="sm"
                        p="md"
                        bg={isDark ? 'dark.6' : 'gray.0'}
                        style={{
                            minHeight: '400px',
                            transition: 'background-color 200ms ease'
                        }}
                    >
                        {renderPlot(plotData, data.visualization_metadata[key])}
                    </Paper>
                </Grid.Col>
            ))}
        </Grid>
    );
}

export default VisualizationDashboard; 