#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QComboBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QScrollArea>
#include <QStatusBar>
#include <QImage>
#include <QPixmap>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

// First native C++ port of AutoRetouch v7. Local-only processing.
// Face mask uses OpenCV Haar cascade as a dependency-light fallback; it is not yet
// pixel-identical to the original MediaPipe FaceMesh + Selfie Segmentation pipeline.

static cv::Mat toDisplayBgr(const cv::Mat& src) {
    if (src.empty()) return {};
    cv::Mat out;
    if (src.channels() == 4) cv::cvtColor(src, out, cv::COLOR_BGRA2BGR);
    else if (src.channels() == 1) cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
    else out = src.clone();
    return out;
}

static cv::Mat makeSkinMask(const cv::Mat& bgr, QString& message) {
    cv::Mat mask = cv::Mat::zeros(bgr.size(), CV_32F);
    cv::CascadeClassifier cascade;
    QStringList candidates;
    const QByteArray env = qgetenv("OPENCV_HAAR_CASCADE");
    if (!env.isEmpty()) candidates << QString::fromLocal8Bit(env);
    candidates << QDir(QCoreApplication::applicationDirPath()).filePath("haarcascade_frontalface_default.xml")
               << QDir(QCoreApplication::applicationDirPath()).filePath("data/haarcascade_frontalface_default.xml")
               << "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";
    bool loaded = false;
    for (const QString& p : candidates) if (QFileInfo::exists(p) && cascade.load(p.toStdString())) { loaded = true; break; }
    if (!loaded) {
        message = "Face detector not found. Put haarcascade_frontalface_default.xml beside the EXE or set OPENCV_HAAR_CASCADE.";
        return mask;
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    std::vector<cv::Rect> faces;
    cascade.detectMultiScale(gray, faces, 1.1, 4, cv::CASCADE_SCALE_IMAGE, cv::Size(30,30));
    if (faces.empty()) { message = "No face detected. You can still open and save the image; retouch mask is empty."; return mask; }
    for (const auto& r : faces) {
        // Approximate face oval and exclude eye/mouth bands; v7's FaceMesh masks will replace this fallback.
        cv::Point center(r.x + r.width/2, r.y + int(r.height*0.52));
        cv::Size axes(std::max(1, int(r.width*0.43)), std::max(1, int(r.height*0.48)));
        cv::Mat local = cv::Mat::zeros(bgr.size(), CV_8U);
        cv::ellipse(local, center, axes, 0, 0, 360, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
        cv::Rect eyeBand(r.x + int(r.width*.10), r.y + int(r.height*.27), int(r.width*.80), std::max(1,int(r.height*.22)));
        cv::Rect mouthBand(r.x + int(r.width*.22), r.y + int(r.height*.68), int(r.width*.56), std::max(1,int(r.height*.16)));
        eyeBand &= cv::Rect(0,0,bgr.cols,bgr.rows); mouthBand &= cv::Rect(0,0,bgr.cols,bgr.rows);
        if (eyeBand.area()>0) local(eyeBand).setTo(0);
        if (mouthBand.area()>0) local(mouthBand).setTo(0);
        cv::Mat f; local.convertTo(f, CV_32F, 1.0/255.0);
        cv::max(mask, f, mask);
    }
    cv::GaussianBlur(mask, mask, cv::Size(0,0), std::max(1.0, std::max(bgr.cols,bgr.rows)/700.0));
    message = QString("Detected %1 face(s). Approximate skin mask ready.").arg(faces.size());
    return mask;
}

static cv::Mat renderImage(const cv::Mat& original, const cv::Mat& mask,
                           int strength, int blend, int warmth, int texture,
                           int exposure, int contrast, int saturation, int blurAmount) {
    if (original.empty()) return {};
    cv::Mat src; original.convertTo(src, CV_32FC3, 1.0/255.0);
    cv::Mat lab; cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> c; cv::split(lab, c);
    cv::Mat smooth; cv::GaussianBlur(c[0], smooth, cv::Size(0,0), std::max(1.0, std::max(original.cols,original.rows)/180.0));
    const float amount = (strength/100.0f) * (blend/100.0f);
    cv::Mat correction = (smooth - c[0]) * amount;
    // Texture 100 preserves original micro-texture; lower values soften, higher values reinforce it.
    cv::Mat detail = c[0] - smooth;
    correction += detail * ((texture-100)/100.0f);
    c[0] += correction.mul(mask);
    c[0] += (exposure / 100.0f) * 18.0f;
    cv::Mat centered = c[0] - 50.0f;
    c[0] = centered * (1.0f + contrast/100.0f) + 50.0f;
    c[1] += warmth * 0.035f * mask;
    c[2] += warmth * 0.14f * mask;
    cv::merge(c, lab);
    cv::Mat out; cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
    // Global saturation adjustment in HSV.
    if (saturation != 0) {
        cv::Mat hsv; cv::cvtColor(out, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> h; cv::split(hsv,h);
        h[1] *= (1.0 + saturation/100.0);
        cv::merge(h,hsv); cv::cvtColor(hsv,out,cv::COLOR_HSV2BGR);
    }
    if (blurAmount > 0) {
        cv::Mat soft; cv::GaussianBlur(out,soft,cv::Size(0,0), 0.3 + blurAmount/15.0);
        cv::addWeighted(out, 1.0-blurAmount/100.0, soft, blurAmount/100.0, 0, out);
    }
    cv::max(out, 0, out); cv::min(out, 1, out);
    cv::Mat u8; out.convertTo(u8,CV_8UC3,255.0);
    return u8;
}

static QImage matToQImage(const cv::Mat& bgr) {
    if (bgr.empty()) return {};
    cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, int(rgb.step), QImage::Format_RGB888).copy();
}

class MainWindow final : public QMainWindow {
public:
    MainWindow() {
        setWindowTitle("Auto Retouch - Smart Dodge & Burn (C++ port)");
        resize(1460, 880); setMinimumSize(1000,640);
        auto* root = new QWidget(this); auto* outer = new QVBoxLayout(root);
        auto* top = new QHBoxLayout();
        auto* open = new QPushButton("Open"); auto* close = new QPushButton("Close image");
        auto* save = new QPushButton("Save retouched");
        top->addWidget(open); top->addWidget(close); top->addWidget(save); top->addStretch();
        viewBox = new QComboBox(); viewBox->addItems({"Preview","Original","Mask"}); top->addWidget(new QLabel("View:")); top->addWidget(viewBox);
        outer->addLayout(top);
        auto* body = new QHBoxLayout();
        imageLabel = new QLabel("Open a portrait to begin"); imageLabel->setAlignment(Qt::AlignCenter);
        imageLabel->setMinimumSize(500,400); imageLabel->setStyleSheet("QLabel { background:#202020; color:#dddddd; }");
        imageLabel->setSizePolicy(QSizePolicy::Ignored,QSizePolicy::Ignored);
        auto* scroll = new QScrollArea(); scroll->setWidget(imageLabel); scroll->setWidgetResizable(true); body->addWidget(scroll,1);
        auto* panel = new QWidget(); panel->setMinimumWidth(300); panel->setMaximumWidth(360); auto* pv = new QVBoxLayout(panel);
        auto* modes = new QGroupBox("Detection mode"); auto* mv = new QVBoxLayout(modes); modeBox = new QComboBox(); modeBox->addItems({"Auto","Face","Half-body","Full-body"}); mv->addWidget(modeBox); pv->addWidget(modes);
        auto* retouch = new QGroupBox("Skin retouch"); auto* rf = new QFormLayout(retouch);
        addSlider(rf,"Strength",strength,0,100,100); addSlider(rf,"Blend",blend,0,100,100); addSlider(rf,"Warmth",warmth,-100,100,0); addSlider(rf,"Texture",texture,0,200,100);
        pv->addWidget(retouch);
        auto* adjust = new QGroupBox("Adjust (whole image)"); auto* af = new QFormLayout(adjust);
        addSlider(af,"Exposure",exposure,-100,100,0); addSlider(af,"Contrast",contrast,-100,100,0); addSlider(af,"Saturation",saturation,-100,100,0); addSlider(af,"Blur",blurAmount,0,100,0);
        pv->addWidget(adjust); pv->addStretch();
        auto* hint = new QLabel("C++ migration starter\nProcessing stays on this PC."); hint->setWordWrap(true); pv->addWidget(hint);
        body->addWidget(panel); outer->addLayout(body,1); setCentralWidget(root);
        statusBar()->showMessage("Ready. Open an image.");
        connect(open,&QPushButton::clicked,this,[this]{openImage();});
        connect(close,&QPushButton::clicked,this,[this]{original.release(); mask.release(); preview.release(); currentPath.clear(); imageLabel->setPixmap(QPixmap()); imageLabel->setText("Open a portrait to begin"); statusBar()->showMessage("Image closed.");});
        connect(save,&QPushButton::clicked,this,[this]{saveImage();});
        connect(viewBox,&QComboBox::currentTextChanged,this,[this]{refreshView();});
        for (QSlider* s : sliders) connect(s,&QSlider::valueChanged,this,[this]{recompute();});
        connect(modeBox,&QComboBox::currentTextChanged,this,[this]{if(!original.empty()) { QString msg; mask=makeSkinMask(original,msg); recompute(); statusBar()->showMessage(msg); }});
    }
private:
    QLabel* imageLabel{}; QComboBox* viewBox{}; QComboBox* modeBox{};
    cv::Mat original,mask,preview; QString currentPath; std::vector<QSlider*> sliders;
    QSlider *strength{},*blend{},*warmth{},*texture{},*exposure{},*contrast{},*saturation{},*blurAmount{};
    void addSlider(QFormLayout* form,const QString& label,QSlider*& target,int lo,int hi,int initial) {
        auto* row = new QWidget(); auto* h = new QHBoxLayout(row); h->setContentsMargins(0,0,0,0);
        target = new QSlider(Qt::Horizontal); target->setRange(lo,hi); target->setValue(initial);
        auto* val = new QLabel(QString::number(initial)); val->setMinimumWidth(36); val->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
        connect(target,&QSlider::valueChanged,val,[val](int v){val->setText(QString::number(v));});
        h->addWidget(target,1); h->addWidget(val); form->addRow(label,row); sliders.push_back(target);
    }
    void openImage() {
        const QString path=QFileDialog::getOpenFileName(this,"Open image",QString(),"Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)");
        if(path.isEmpty()) return;
        cv::Mat loaded=cv::imread(path.toLocal8Bit().constData(),cv::IMREAD_UNCHANGED);
        if(loaded.empty()){QMessageBox::critical(this,"Open failed","The selected image could not be opened.");return;}
        original=toDisplayBgr(loaded); currentPath=path;
        QString msg; mask=makeSkinMask(original,msg); recompute(); statusBar()->showMessage(msg);
    }
    void recompute() {
        if(original.empty()) return;
        preview=renderImage(original,mask,strength->value(),blend->value(),warmth->value(),texture->value(),exposure->value(),contrast->value(),saturation->value(),blurAmount->value());
        refreshView();
    }
    void refreshView() {
        if(original.empty()) return;
        cv::Mat shown;
        if(viewBox->currentText()=="Original") shown=original;
        else if(viewBox->currentText()=="Mask") { cv::Mat m; mask.convertTo(m,CV_8U,255); cv::cvtColor(m,shown,cv::COLOR_GRAY2BGR); }
        else shown=preview;
        QPixmap pix=QPixmap::fromImage(matToQImage(shown));
        imageLabel->setPixmap(pix.scaled(imageLabel->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation));
    }
    void resizeEvent(QResizeEvent* e) override { QMainWindow::resizeEvent(e); refreshView(); }
    void saveImage() {
        if(preview.empty()||currentPath.isEmpty()){QMessageBox::information(this,"Save","Open an image first.");return;}
        QFileInfo fi(currentPath); QString out=fi.dir().filePath(fi.completeBaseName()+"_retouched."+fi.suffix());
        if(QFileInfo::exists(out)) out=fi.dir().filePath(fi.completeBaseName()+"_retouched_2."+fi.suffix());
        if(!cv::imwrite(out.toLocal8Bit().constData(),preview)){QMessageBox::critical(this,"Save failed","Could not save output image.");return;}
        statusBar()->showMessage("Saved: "+out); QMessageBox::information(this,"Saved","Saved retouched image to:\n"+out);
    }
};

int main(int argc,char** argv) {
    QApplication app(argc,argv);
    QApplication::setApplicationName("AutoRetouchCpp");
    MainWindow window; window.show();
    return app.exec();
}
