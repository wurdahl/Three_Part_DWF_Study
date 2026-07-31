#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

inline void write_two_series_svg(
    const std::string& filename,
    const std::string& title,
    const std::string& ylabel,
    const std::vector<double>& y1,
    const std::vector<double>& y2,
    const std::string& label1,
    const std::string& label2,
    bool log_y)
{
    const int W = 900;
    const int H = 560;
    const double L = 90.0;
    const double R = 35.0;
    const double T = 60.0;
    const double B = 70.0;
    const double PW = W - L - R;
    const double PH = H - T - B;

    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();

    auto scan = [&](const std::vector<double>& values)
    {
        for (double value : values)
        {
            double plotted = std::abs(value);

            if (log_y)
            {
                if (plotted <= 0.0)
                    continue;

                plotted = std::log10(plotted);
            }

            ymin = std::min(ymin, plotted);
            ymax = std::max(ymax, plotted);
        }
    };

    scan(y1);
    scan(y2);

    if (!(ymax > ymin))
    {
        ymin -= 0.5;
        ymax += 0.5;
    }

    const double padding = 0.08 * (ymax - ymin);
    ymin -= padding;
    ymax += padding;

    auto sx = [&](double x)
    {
        return L + PW * x / (y1.size() - 1.0);
    };

    auto sy = [&](double y)
    {
        double plotted = std::abs(y);

        if (log_y)
            plotted = std::log10(plotted);

        return T + PH * (ymax - plotted) / (ymax - ymin);
    };

    std::ofstream out(filename);

    out << "<svg xmlns='http://www.w3.org/2000/svg' width='"
        << W << "' height='" << H << "'>"
        << "<rect width='100%' height='100%' fill='white'/>"
        << "<style>"
        << "text{font-family:sans-serif}"
        << ".axis{stroke:black;stroke-width:2}"
        << ".grid{stroke:#ddd}"
        << ".one{stroke:#1f77b4;fill:none;stroke-width:2}"
        << ".two{stroke:#ff7f0e;fill:none;stroke-width:2}"
        << "</style>"
        << "<text x='" << W / 2
        << "' y='32' text-anchor='middle' font-size='23'>"
        << title << "</text>";

    for (int i = 0; i <= 5; ++i)
    {
        const double fraction = i / 5.0;
        const double yy = T + PH * fraction;
        const double transformed =
            ymax - fraction * (ymax - ymin);
        const double label =
            log_y ? std::pow(10.0, transformed) : transformed;

        out << "<line class='grid' x1='" << L
            << "' x2='" << L + PW
            << "' y1='" << yy
            << "' y2='" << yy << "'/>"
            << "<text x='" << L - 8
            << "' y='" << yy + 5
            << "' text-anchor='end' font-size='13'>"
            << label << "</text>";
    }

    out << "<line class='axis' x1='" << L
        << "' x2='" << L + PW
        << "' y1='" << T + PH
        << "' y2='" << T + PH << "'/>"
        << "<line class='axis' x1='" << L
        << "' x2='" << L
        << "' y1='" << T
        << "' y2='" << T + PH << "'/>";

    out << "<polyline class='one' points='";

    for (std::size_t i = 0; i < y1.size(); ++i)
        out << sx(i) << "," << sy(y1[i]) << " ";

    out << "'/><polyline class='two' points='";

    for (std::size_t i = 0; i < y2.size(); ++i)
        out << sx(i) << "," << sy(y2[i]) << " ";

    out << "'/>"
        << "<text x='" << L + PW / 2
        << "' y='" << H - 20
        << "' text-anchor='middle' font-size='18'>t</text>"
        << "<text x='22' y='" << T + PH / 2
        << "' text-anchor='middle' font-size='18' "
        << "transform='rotate(-90 22 " << T + PH / 2
        << ")'>" << ylabel << "</text>"
        << "<line class='one' x1='" << W - 250
        << "' x2='" << W - 210
        << "' y1='75' y2='75'/>"
        << "<text x='" << W - 200
        << "' y='80' font-size='15'>" << label1 << "</text>"
        << "<line class='two' x1='" << W - 250
        << "' x2='" << W - 210
        << "' y1='100' y2='100'/>"
        << "<text x='" << W - 200
        << "' y='105' font-size='15'>" << label2 << "</text>"
        << "</svg>";
}

inline void write_single_series_svg(
    const std::string& filename,
    const std::string& title,
    const std::string& ylabel,
    const std::vector<double>& values,
    double horizontal_line =
        std::numeric_limits<double>::quiet_NaN(),
    int vertical_line = -1)
{
    const int W = 900;
    const int H = 560;
    const double L = 90.0;
    const double R = 35.0;
    const double T = 60.0;
    const double B = 70.0;
    const double PW = W - L - R;
    const double PH = H - T - B;

    double ymin =
        *std::min_element(values.begin(), values.end());
    double ymax =
        *std::max_element(values.begin(), values.end());

    if (std::isfinite(horizontal_line))
    {
        ymin = std::min(ymin, horizontal_line);
        ymax = std::max(ymax, horizontal_line);
    }

    if (!(ymax > ymin))
    {
        ymin -= 0.5;
        ymax += 0.5;
    }

    const double padding = 0.08 * (ymax - ymin);
    ymin -= padding;
    ymax += padding;

    auto sx = [&](double x)
    {
        return L + PW * x / (values.size() - 1.0);
    };

    auto sy = [&](double y)
    {
        return T + PH * (ymax - y) / (ymax - ymin);
    };

    std::ofstream out(filename);

    out << "<svg xmlns='http://www.w3.org/2000/svg' width='"
        << W << "' height='" << H << "'>"
        << "<rect width='100%' height='100%' fill='white'/>"
        << "<style>"
        << "text{font-family:sans-serif}"
        << ".axis{stroke:black;stroke-width:2}"
        << ".grid{stroke:#ddd}"
        << ".data{stroke:#1f77b4;fill:none;stroke-width:2}"
        << ".ref{stroke:#ff7f0e;stroke-width:2;stroke-dasharray:7 5}"
        << "</style>"
        << "<text x='" << W / 2
        << "' y='32' text-anchor='middle' font-size='23'>"
        << title << "</text>";

    for (int i = 0; i <= 5; ++i)
    {
        const double fraction = i / 5.0;
        const double yy = T + PH * fraction;
        const double label =
            ymax - fraction * (ymax - ymin);

        out << "<line class='grid' x1='" << L
            << "' x2='" << L + PW
            << "' y1='" << yy
            << "' y2='" << yy << "'/>"
            << "<text x='" << L - 8
            << "' y='" << yy + 5
            << "' text-anchor='end' font-size='13'>"
            << label << "</text>";
    }

    out << "<line class='axis' x1='" << L
        << "' x2='" << L + PW
        << "' y1='" << T + PH
        << "' y2='" << T + PH << "'/>"
        << "<line class='axis' x1='" << L
        << "' x2='" << L
        << "' y1='" << T
        << "' y2='" << T + PH << "'/>"
        << "<polyline class='data' points='";

    for (std::size_t i = 0; i < values.size(); ++i)
        out << sx(i) << "," << sy(values[i]) << " ";

    out << "'/>";

    if (std::isfinite(horizontal_line))
    {
        out << "<line class='ref' x1='" << L
            << "' x2='" << L + PW
            << "' y1='" << sy(horizontal_line)
            << "' y2='" << sy(horizontal_line) << "'/>";
    }

    if (vertical_line >= 0)
    {
        out << "<line class='ref' x1='" << sx(vertical_line)
            << "' x2='" << sx(vertical_line)
            << "' y1='" << T
            << "' y2='" << T + PH << "'/>";
    }

    out << "<text x='" << L + PW / 2
        << "' y='" << H - 20
        << "' text-anchor='middle' font-size='18'>"
        << (vertical_line >= 0 ? "trajectory" : "t")
        << "</text>"
        << "<text x='22' y='" << T + PH / 2
        << "' text-anchor='middle' font-size='18' "
        << "transform='rotate(-90 22 " << T + PH / 2
        << ")'>" << ylabel << "</text>"
        << "</svg>";
}
