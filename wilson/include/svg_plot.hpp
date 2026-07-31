#pragma once

#include "parameters.hpp"
#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

inline void write_svg_plot(
    const std::string& filename,
    const std::string& title,
    const std::string& y_label,
    const Corr& y,
    const Corr& y_error,
    bool logarithmic_y,
    double horizontal_line = std::numeric_limits<double>::quiet_NaN(),
    int plateau_first_t = -1,
    int plateau_last_t = -1)
{
    constexpr int width = 1000;
    constexpr int height = 600;
    constexpr double left = 100.0;
    constexpr double right = 40.0;
    constexpr double top = 70.0;
    constexpr double bottom = 80.0;

    const double plot_width = width - left - right;
    const double plot_height = height - top - bottom;

    std::vector<int> valid_times;
    for (int t = 0; t < Nt; ++t)
    {
        if (!std::isfinite(y[t]) || !std::isfinite(y_error[t]))
            continue;
        if (logarithmic_y && y[t] <= 0.0)
            continue;
        valid_times.push_back(t);
    }

    if (valid_times.empty())
        throw std::runtime_error("No valid points for " + filename);

    double y_min = std::numeric_limits<double>::infinity();
    double y_max = -std::numeric_limits<double>::infinity();

    for (int t : valid_times)
    {
        double low = y[t] - y_error[t];
        double high = y[t] + y_error[t];
        if (logarithmic_y)
        {
            low = std::max(low, 0.5 * y[t]);
            y_min = std::min(y_min, std::log10(low));
            y_max = std::max(y_max, std::log10(std::max(high, y[t])));
        }
        else
        {
            y_min = std::min(y_min, low);
            y_max = std::max(y_max, high);
        }
    }

    if (std::isfinite(horizontal_line))
    {
        const double transformed_line =
            logarithmic_y ? std::log10(horizontal_line) : horizontal_line;
        y_min = std::min(y_min, transformed_line);
        y_max = std::max(y_max, transformed_line);
    }

    if (!(y_max > y_min))
    {
        y_min -= 0.5;
        y_max += 0.5;
    }

    const double padding = 0.08 * (y_max - y_min);
    y_min -= padding;
    y_max += padding;

    auto x_to_svg = [&](double x)
    {
        return left + plot_width * x / (Nt - 1.0);
    };

    auto y_to_svg = [&](double value)
    {
        const double transformed = logarithmic_y ? std::log10(value) : value;
        return top + plot_height * (y_max - transformed) / (y_max - y_min);
    };

    std::ofstream out(filename);
    out << "<svg xmlns='http://www.w3.org/2000/svg' width='" << width
        << "' height='" << height << "' viewBox='0 0 " << width << ' '
        << height << "'>\n"
        << "<rect width='100%' height='100%' fill='white'/>\n"
        << "<style>text{font-family:sans-serif;fill:black}"
        << ".axis{stroke:black;stroke-width:2}"
        << ".grid{stroke:#dddddd;stroke-width:1}"
        << ".data{stroke:#1f77b4;fill:#1f77b4}"
        << ".fit{stroke:#d62728;stroke-width:2;stroke-dasharray:8 6}"
        << ".window{fill:#dddddd;opacity:0.45}</style>\n"
        << "<text x='" << width / 2 << "' y='36' font-size='24' "
        << "text-anchor='middle'>" << title << "</text>\n";

    if (plateau_first_t >= 0 && plateau_last_t >= plateau_first_t)
    {
        const double x1 = x_to_svg(plateau_first_t - 0.45);
        const double x2 = x_to_svg(plateau_last_t + 0.45);
        out << "<rect class='window' x='" << x1 << "' y='" << top
            << "' width='" << x2 - x1 << "' height='" << plot_height
            << "'/>\n";
    }

    for (int i = 0; i <= 5; ++i)
    {
        const double fraction = i / 5.0;
        const double svg_y = top + plot_height * fraction;
        const double transformed = y_max - fraction * (y_max - y_min);
        const double label = logarithmic_y ? std::pow(10.0, transformed)
                                           : transformed;
        out << "<line class='grid' x1='" << left << "' x2='"
            << left + plot_width << "' y1='" << svg_y << "' y2='"
            << svg_y << "'/>\n"
            << "<text x='" << left - 12 << "' y='" << svg_y + 5
            << "' font-size='14' text-anchor='end'>" << label
            << "</text>\n";
    }

    const int x_tick_spacing = std::max(1, Nt / 10);
    for (int t = 0; t < Nt; t += x_tick_spacing)
    {
        const double svg_x = x_to_svg(t);
        out << "<line class='grid' x1='" << svg_x << "' x2='" << svg_x
            << "' y1='" << top << "' y2='" << top + plot_height
            << "'/>\n"
            << "<text x='" << svg_x << "' y='" << top + plot_height + 27
            << "' font-size='14' text-anchor='middle'>" << t
            << "</text>\n";
    }

    out << "<line class='axis' x1='" << left << "' x2='"
        << left + plot_width << "' y1='" << top + plot_height << "' y2='"
        << top + plot_height << "'/>\n"
        << "<line class='axis' x1='" << left << "' x2='" << left
        << "' y1='" << top << "' y2='" << top + plot_height << "'/>\n"
        << "<text x='" << left + plot_width / 2 << "' y='" << height - 24
        << "' font-size='19' text-anchor='middle'>t</text>\n"
        << "<text x='25' y='" << top + plot_height / 2
        << "' font-size='19' text-anchor='middle' transform='rotate(-90 25 "
        << top + plot_height / 2 << ")'>" << y_label << "</text>\n";

    if (std::isfinite(horizontal_line))
    {
        const double svg_y = y_to_svg(horizontal_line);
        out << "<line class='fit' x1='" << left << "' x2='"
            << left + plot_width << "' y1='" << svg_y << "' y2='"
            << svg_y << "'/>\n"
            << "<text x='" << left + plot_width - 8 << "' y='" << svg_y - 8
            << "' font-size='15' text-anchor='end'>selected mass = "
            << horizontal_line << "</text>\n";
    }

    for (int t : valid_times)
    {
        const double svg_x = x_to_svg(t);
        double low = y[t] - y_error[t];
        const double high = y[t] + y_error[t];
        if (logarithmic_y)
            low = std::max(low, 0.5 * y[t]);

        const double svg_low = y_to_svg(low);
        const double svg_high = y_to_svg(high);
        const double svg_center = y_to_svg(y[t]);

        out << "<line class='data' x1='" << svg_x << "' x2='" << svg_x
            << "' y1='" << svg_high << "' y2='" << svg_low << "'/>\n"
            << "<line class='data' x1='" << svg_x - 5 << "' x2='"
            << svg_x + 5 << "' y1='" << svg_high << "' y2='" << svg_high
            << "'/>\n"
            << "<line class='data' x1='" << svg_x - 5 << "' x2='"
            << svg_x + 5 << "' y1='" << svg_low << "' y2='" << svg_low
            << "'/>\n"
            << "<circle class='data' cx='" << svg_x << "' cy='"
            << svg_center << "' r='4'/>\n";
    }

    out << "</svg>\n";
}


inline void write_gauge_history_svg(const std::string& filename,
                                    const std::vector<double>& plaquettes,
                                    int thermalization)
{
    if(plaquettes.empty()) return;
    constexpr int W=1000,H=600;
    constexpr double L=100,R=40,T=70,B=80;
    double ymin=*std::min_element(plaquettes.begin(),plaquettes.end());
    double ymax=*std::max_element(plaquettes.begin(),plaquettes.end());
    if(!(ymax>ymin)){ymin-=0.5;ymax+=0.5;}
    double pad=0.08*(ymax-ymin);ymin-=pad;ymax+=pad;
    auto sx=[&](double x){return L+(W-L-R)*x/(plaquettes.size()-1.0);};
    auto sy=[&](double y){return T+(H-T-B)*(ymax-y)/(ymax-ymin);};
    std::ofstream out(filename);
    out<<"<svg xmlns='http://www.w3.org/2000/svg' width='"<<W
       <<"' height='"<<H<<"'><rect width='100%' height='100%' fill='white'/>"
       <<"<style>text{font-family:sans-serif}.axis{stroke:black;stroke-width:2}"
       <<".data{stroke:#1f77b4;fill:none;stroke-width:2}"
       <<".cut{stroke:#d62728;stroke-dasharray:8 6;stroke-width:2}</style>"
       <<"<text x='"<<W/2<<"' y='36' font-size='24' text-anchor='middle'>"
       <<"Wilson HMC plaquette history</text>";
    out<<"<line class='axis' x1='"<<L<<"' x2='"<<W-R<<"' y1='"<<H-B
       <<"' y2='"<<H-B<<"'/><line class='axis' x1='"<<L<<"' x2='"<<L
       <<"' y1='"<<T<<"' y2='"<<H-B<<"'/>";
    out<<"<polyline class='data' points='";
    for(size_t i=0;i<plaquettes.size();++i)
        out<<sx(i)<<","<<sy(plaquettes[i])<<" ";
    out<<"'/>";
    if(thermalization>=0&&thermalization<(int)plaquettes.size())
        out<<"<line class='cut' x1='"<<sx(thermalization)<<"' x2='"
           <<sx(thermalization)<<"' y1='"<<T<<"' y2='"<<H-B<<"'/>";
    out<<"<text x='"<<W/2<<"' y='"<<H-20
       <<"' text-anchor='middle' font-size='18'>trajectory</text>"
       <<"<text x='25' y='"<<H/2
       <<"' text-anchor='middle' font-size='18' transform='rotate(-90 25 "
       <<H/2<<")'>average plaquette</text></svg>";
}
