// BEN 2026-11-01
// Change aim: C++ port of python/recon_lib.py: freehand 3D ultrasound reconstruction and evaluation on a simulated sweep.
// File: reconstruct.cpp
// SIMULATED data only. Units: millimetres. Usage: reconstruct --data demo_data --out results --tag cpp
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static const double kProstateThr = 0.475;
static const double kLesionThr = 0.425;
static const double kRoiHalf[3] = {28.0, 24.0, 20.0};
static const double kLesionRoiRadius = 6.0;

static std::string getArg(int argc, char** argv, const std::string& name, const std::string& def)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (name == argv[i]) return argv[i + 1];
    return def;
}

static std::map<std::string, double> loadMeta(const std::string& dir)
{
    std::map<std::string, double> m;
    std::ifstream in(dir + "/meta.txt");
    std::string line;
    while (std::getline(in, line)) {
        const size_t eq = line.find('=');
        if (eq != std::string::npos) m[line.substr(0, eq)] = std::stod(line.substr(eq + 1));
    }
    return m;
}

static std::vector<float> readFloats(const std::string& path, size_t count)
{
    std::vector<float> v(count);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) { std::cerr << "cannot read " << path << "\n"; std::exit(1); }
    return v;
}

// 3x3x3 sum (zero padded) of a volume stored as [k][j][i]
static std::vector<double> boxSum(const std::vector<double>& a, int n)
{
    std::vector<double> out(a.size(), 0.0);
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int kk = k + dz, jj = j + dy, ii = i + dx;
                            if (kk < 0 || kk >= n || jj < 0 || jj >= n || ii < 0 || ii >= n) continue;
                            s += a[(static_cast<size_t>(kk) * n + jj) * n + ii];
                        }
                out[(static_cast<size_t>(k) * n + j) * n + i] = s;
            }
    return out;
}

int main(int argc, char** argv)
{
    const std::string dir = getArg(argc, argv, "--data", "demo_data");
    const std::string outDir = getArg(argc, argv, "--out", "results");
    const std::string tag = getArg(argc, argv, "--tag", "cpp");
    std::map<std::string, double> m = loadMeta(dir);
    const int n = static_cast<int>(m["N"]), S = static_cast<int>(m["S"]), H = static_cast<int>(m["H"]), W = static_cast<int>(m["W"]);
    const double dx = m["dx"], pdx = m["pdx"];
    const size_t n3 = static_cast<size_t>(n) * n * n;

    const std::vector<float> slices = readFloats(dir + "/slices.bin", static_cast<size_t>(S) * H * W);
    const std::vector<float> gtF = readFloats(dir + "/gt_volume.bin", n3);
    std::vector<double> gt(gtF.begin(), gtF.end());
    std::vector<std::vector<double>> poses;
    {
        std::ifstream in(dir + "/poses_est.csv");
        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) {
            std::stringstream ss(line);
            std::string tok;
            std::vector<double> row;
            while (std::getline(ss, tok, ',')) row.push_back(std::stod(tok));
            if (row.size() == 12) poses.push_back(row);
        }
    }
    if (static_cast<int>(poses.size()) != S) { std::cerr << "pose count mismatch\n"; return 1; }

    // pixel binning
    std::vector<double> total(n3, 0.0), count(n3, 0.0);
    long long used = 0;
    for (int s = 0; s < S; ++s) {
        const std::vector<double>& p = poses[s];
        for (int v = 0; v < H; ++v)
            for (int u = 0; u < W; ++u) {
                const double x = (u + 0.5 - W / 2.0) * pdx, y = (v + 0.5 - H / 2.0) * pdx;
                const double px = x * p[0] + y * p[1] + p[9];
                const double py = x * p[3] + y * p[4] + p[10];
                const double pz = x * p[6] + y * p[7] + p[11];
                const long long ii = static_cast<long long>(std::floor(px / dx + n / 2.0));
                const long long jj = static_cast<long long>(std::floor(py / dx + n / 2.0));
                const long long kk = static_cast<long long>(std::floor(pz / dx + n / 2.0));
                if (ii < 0 || ii >= n || jj < 0 || jj >= n || kk < 0 || kk >= n) continue;
                const size_t f = (static_cast<size_t>(kk) * n + jj) * n + ii;
                total[f] += slices[(static_cast<size_t>(s) * H + v) * W + u];
                count[f] += 1.0;
                ++used;
            }
    }
    std::vector<double> vol(n3, 0.0);
    std::vector<char> filled(n3, 0);
    size_t nDirect = 0;
    for (size_t f = 0; f < n3; ++f)
        if (count[f] > 0) { vol[f] = total[f] / count[f]; filled[f] = 1; ++nDirect; }

    // hole filling (3 passes of 3x3x3 neighbour averaging)
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<double> v0(n3), c0(n3);
        for (size_t f = 0; f < n3; ++f) { v0[f] = filled[f] ? vol[f] : 0.0; c0[f] = filled[f] ? 1.0 : 0.0; }
        const std::vector<double> s3 = boxSum(v0, n), c3 = boxSum(c0, n);
        for (size_t f = 0; f < n3; ++f)
            if (!filled[f] && c3[f] > 0) { vol[f] = s3[f] / c3[f]; filled[f] = 2; }
        for (size_t f = 0; f < n3; ++f) if (filled[f] == 2) filled[f] = 1;
    }
    size_t nFilled = 0;
    for (size_t f = 0; f < n3; ++f) nFilled += filled[f] ? 1 : 0;

    // evaluation
    std::vector<double> ones(n3, 1.0);
    const std::vector<double> sv = boxSum(vol, n), sc = boxSum(ones, n);
    std::vector<double> sm(n3);
    for (size_t f = 0; f < n3; ++f) sm[f] = sv[f] / sc[f];

    const double lc[3] = {m["lx"], m["ly"], m["lz"]};
    const double nom[3] = {lc[0] + m["nx"], lc[1] + m["ny"], lc[2] + m["nz"]};
    double sa = 0, sb = 0, nRoi = 0;
    std::vector<size_t> roiIdx;
    long long pBoth = 0, pRec = 0, pGt = 0, lBoth = 0, lRec = 0, lGt = 0;
    double cx = 0, cy = 0, cz = 0;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const size_t f = (static_cast<size_t>(k) * n + j) * n + i;
                const double x = (i + 0.5 - n / 2.0) * dx, y = (j + 0.5 - n / 2.0) * dx, z = (k + 0.5 - n / 2.0) * dx;
                if (std::fabs(x) <= kRoiHalf[0] && std::fabs(y) <= kRoiHalf[1] && std::fabs(z) <= kRoiHalf[2]) {
                    roiIdx.push_back(f);
                    sa += vol[f]; sb += gt[f]; nRoi += 1;
                }
                const bool pr = sm[f] > kProstateThr, pg = gt[f] > kProstateThr;
                pRec += pr; pGt += pg; pBoth += (pr && pg);
                const double d2 = (x - nom[0]) * (x - nom[0]) + (y - nom[1]) * (y - nom[1]) + (z - nom[2]) * (z - nom[2]);
                const bool inBox = d2 <= kLesionRoiRadius * kLesionRoiRadius;
                const bool lr = (sm[f] < kLesionThr) && inBox && filled[f], lg = (gt[f] < kLesionThr) && inBox;
                lRec += lr; lGt += lg; lBoth += (lr && lg);
                if (lr) { cx += x; cy += y; cz += z; }
            }
    const double ma = sa / nRoi, mb = sb / nRoi;
    double sab = 0, saa = 0, sbb = 0, sse = 0;
    for (size_t f : roiIdx) {
        const double da = vol[f] - ma, db = gt[f] - mb;
        sab += da * db; saa += da * da; sbb += db * db;
        sse += (vol[f] - gt[f]) * (vol[f] - gt[f]);
    }
    const double ncc = sab / std::sqrt(saa * sbb), rmse = std::sqrt(sse / nRoi);
    const double pDice = 2.0 * pBoth / static_cast<double>(pRec + pGt);
    const double pVol = 100.0 * (static_cast<double>(pRec) - static_cast<double>(pGt)) / static_cast<double>(pGt);
    const double lDice = 2.0 * lBoth / static_cast<double>(std::max<long long>(lRec + lGt, 1));
    double lErr = std::nan("");
    if (lRec > 0) {
        const double ex = cx / lRec - lc[0], ey = cy / lRec - lc[1], ez = cz / lRec - lc[2];
        lErr = std::sqrt(ex * ex + ey * ey + ez * ez);
    }

    std::ostringstream o;
    o << std::fixed;
    o << "slices: " << S << ", pixels used: " << used << "\n";
    o << std::setprecision(2) << "voxels filled directly: " << 100.0 * nDirect / n3 << " %, after hole filling: " << 100.0 * nFilled / n3 << " %\n";
    o << std::setprecision(4) << "recon vs ground truth in ROI: NCC " << ncc << ", RMSE " << rmse << "\n";
    o << "prostate segmentation: Dice " << pDice << ", volume error " << std::setprecision(2) << pVol << " %\n";
    o << std::setprecision(4) << "lesion: Dice " << lDice << ", centroid error " << std::setprecision(3) << lErr << " mm\n";
    std::cout << o.str();
    std::ofstream(outDir + "/summary_" + tag + ".txt") << o.str();

    std::vector<float> out(vol.begin(), vol.end());
    std::ofstream(outDir + "/recon_" + tag + ".bin", std::ios::binary).write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(n3 * sizeof(float)));

    // three orthogonal slices of the reconstruction through the true lesion centre
    const int li = static_cast<int>(std::floor(lc[0] / dx + n / 2.0)), lj = static_cast<int>(std::floor(lc[1] / dx + n / 2.0)),
              lk = static_cast<int>(std::floor(lc[2] / dx + n / 2.0));
    cv::Mat axial(n, n, CV_8U), coronal(n, n, CV_8U), sagittal(n, n, CV_8U);
    auto g = [&](double v) { return static_cast<uchar>(std::min(255.0, std::max(0.0, v * 255.0 / 0.9))); };
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            axial.at<uchar>(a, b) = g(vol[(static_cast<size_t>(lk) * n + a) * n + b]);       // rows y, cols x
            coronal.at<uchar>(a, b) = g(vol[(static_cast<size_t>(a) * n + lj) * n + b]);     // rows z, cols x
            sagittal.at<uchar>(a, b) = g(vol[(static_cast<size_t>(a) * n + b) * n + li]);    // rows z, cols y
        }
    cv::Mat row, big;
    cv::hconcat(std::vector<cv::Mat>{axial, coronal, sagittal}, row);
    cv::resize(row, big, cv::Size(), 3, 3, cv::INTER_NEAREST);
    cv::cvtColor(big, big, cv::COLOR_GRAY2BGR);
    cv::putText(big, "SIMULATED  C++ reconstruction: axial | coronal | sagittal", cv::Point(6, 18), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::imwrite(outDir + "/recon_slices_" + tag + ".png", big);
    return 0;
}
