// compile with: /D_UNICODE /DUNICODE /DWIN32 /D_WINDOWS /c

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <tchar.h>

#include <chrono>
#include <thread>

# include <opencv2/opencv.hpp>

#include <ATen/ATen.h>
#include <torch/script.h>

#include <codecvt>

// Global variables

// The main window class name.
static TCHAR szWindowClass[] = _T("DesktopApp");

// The string that appears in the application's title bar.
static TCHAR szTitle[] = _T("Detector");

// Stored instance handle for use in Win32 API calls such as FindResource
HINSTANCE hInst;

// Forward declarations of functions included in this code module:
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
DWORD   WINAPI      PaintProc(LPVOID lpParam);

#define MODEL_PATH "detr.pt"

class Detector {
private:
    torch::jit::script::Module model;
    std::string keys[91] = {
     "N/A",
     "person",
     "bicycle",
     "car",
     "motorcycle",
     "airplane",
     "bus",
     "train",
     "truck",
     "boat",
     "traffic light",
     "fire hydrant",
     "N/A",
     "stop sign",
     "parking meter",
     "bench",
     "bird",
     "cat",
     "dog",
     "horse",
     "sheep",
     "cow",
     "elephant",
     "bear",
     "zebra",
     "giraffe",
     "N/A",
     "backpack",
     "umbrella",
     "N/A",
     "N/A",
     "handbag",
     "tie",
     "suitcase",
     "frisbee",
     "skis",
     "snowboard",
     "sports ball",
     "kite",
     "baseball bat",
     "baseball glove",
     "skateboard",
     "surfboard",
     "tennis racket",
     "bottle",
     "N/A",
     "wine glass",
     "cup",
     "fork",
     "knife",
     "spoon",
     "bowl",
     "banana",
     "apple",
     "sandwich",
     "orange",
     "broccoli",
     "carrot",
     "hot dog",
     "pizza",
     "donut",
     "cake",
     "chair",
     "couch",
     "potted plant",
     "bed",
     "N/A",
     "dining table",
     "N/A",
     "N/A",
     "toilet",
     "N/A",
     "tv",
     "laptop",
     "mouse",
     "remote",
     "keyboard",
     "cell phone",
     "microwave",
     "oven",
     "toaster",
     "sink",
     "refrigerator",
     "N/A",
     "book",
     "clock",
     "vase",
     "scissors",
     "teddy bear",
     "hair drier",
     "toothbrush"
    };

    std::vector<torch::jit::IValue> preprocessing(cv::Mat src) {
        cv::Mat reszied_img;
        cv::cvtColor(src, src, cv::COLOR_RGBA2RGB);
        cv::resize(src, reszied_img, cv::Size(800, 800), 0, 0, 1);

        auto input = torch::from_blob(reszied_img.data, { reszied_img.rows, reszied_img.cols, reszied_img.channels() }, torch::kUInt8);

        auto mean = torch::tensor({ { 0.485, 0.456, 0.406 } });
        auto std = torch::tensor({ { 0.229, 0.224, 0.225 } });


        input = input / 255;
        input = (input - mean) / std;

        input = input.permute({ 2, 0, 1 }).unsqueeze(0).type_as(torch::ones({ 1 }));
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        return inputs;
    }

    void draw(cv::Mat src, at::Tensor logits, at::Tensor bboxes) {
        int width = src.cols;
        int height = src.rows;

        for (int i = 0; i < 100; i++) {
            auto cls_idx = logits[i].argmax().item().toInt();

            if (cls_idx == 91) {
                continue;
            }

            auto cls = keys[cls_idx];
            std::cout << cls << "\n";

            float center_x = bboxes[i][0].item().toFloat() * width;
            float center_y = bboxes[i][1].item().toFloat() * height;

            float w = bboxes[i][2].item().toFloat() * width;
            float h = bboxes[i][3].item().toFloat() * height;

            cv::putText(src, cls, cv::Point2f(center_x - (w / 2), center_y - (h / 2)), 0, 1, cv::Scalar(255), 2);
            cv::rectangle(src, cv::Point2f(center_x - (w / 2), center_y - (h / 2)), cv::Point2f(center_x + (w / 2), center_y + (h / 2)), cv::Scalar(255), 2);

        }
    }

public:
    Detector() {
        model = torch::jit::load("detr.pt");
    }

    cv::Mat detect(cv::Mat src) {
        std::vector<torch::jit::IValue> inputs = preprocessing(src);

        try {

            auto output = model.forward(inputs);
            auto logits = output.toTuple()->elements()[0].toTensor()[0];
            auto bboxes = output.toTuple()->elements()[1].toTensor()[0];

            draw(src, logits, bboxes);
            return src;
        }
        catch (const c10::Error& e) {
            std::cerr << e.what();
        }
        
    }
};


Detector detector = Detector();

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR     lpCmdLine,
    _In_ int       nCmdShow
)
{
    WNDCLASSEX wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(wcex.hInstance, IDI_APPLICATION);
    wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(wcex.hInstance, IDI_APPLICATION);

    if (!RegisterClassEx(&wcex))
    {
        MessageBox(NULL,
            _T("Call to RegisterClassEx failed!"),
            _T("Windows Desktop Guided Tour"),
            NULL);

        return 1;
    }

    // Store instance handle in our global variable
    hInst = hInstance;

    // The parameters to CreateWindowEx explained:
    // WS_EX_OVERLAPPEDWINDOW : An optional extended window style.
    // szWindowClass: the name of the application
    // szTitle: the text that appears in the title bar
    // WS_OVERLAPPEDWINDOW: the type of window to create
    // CW_USEDEFAULT, CW_USEDEFAULT: initial position (x, y)
    // 500, 100: initial size (width, length)
    // NULL: the parent of this window
    // NULL: this application does not have a menu bar
    // hInstance: the first parameter from WinMain
    // NULL: not used in this application
    HWND hWnd = CreateWindowEx(
        WS_EX_OVERLAPPEDWINDOW,
        szWindowClass,
        szTitle,
        WS_OVERLAPPEDWINDOW,
        //CW_USEDEFAULT, CW_USEDEFAULT,
        1000, 300,
        500, 500,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    if (!hWnd)
    {
        MessageBox(NULL,
            _T("Call to CreateWindow failed!"),
            _T("Windows Desktop Guided Tour"),
            NULL);

        return 1;
    }

    // The parameters to ShowWindow explained:
    // hWnd: the value returned from CreateWindow
    // nCmdShow: the fourth parameter from WinMain
    ShowWindow(hWnd,
        nCmdShow);
    UpdateWindow(hWnd);

    // Main message loop:
    MSG msg;
    
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return (int)msg.wParam;
}
/*
void screenshot(HDC hdc2)
{
    auto w = GetSystemMetrics(SM_CXFULLSCREEN);
    auto h = GetSystemMetrics(SM_CYFULLSCREEN);
    auto hdc = GetDC(HWND_DESKTOP);
    auto hbitmap = CreateCompatibleBitmap(hdc, w, h);
    auto memdc = CreateCompatibleDC(hdc);
    auto oldbmp = SelectObject(memdc, hbitmap);
    //auto hdcWindow = GetDC(hWnd);
    BitBlt(memdc, 0, 0, w, h, hdc, 0, 0, SRCCOPY);

    cv::Mat mat(h, w, CV_8UC4);
    BITMAPINFOHEADER bi = { sizeof(bi), w, -h, 1, 32, BI_RGB };
    GetDIBits(hdc, hbitmap, 0, h, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    cv::Mat result = detector.detect(mat);

    cv::imshow("image", result);
    cv::waitKey(0);

    auto dc = GetDC(nullptr);
    //HBITMAP bitmap = CreateDIBitmap(dc, nullptr, CBM_INIT, result.data, nullptr, DIB_RGB_COLORS);
    StretchBlt(hdc2,
        0, 0,
        450, 400,
        dc,
        0, 0,
        GetSystemMetrics(SM_CXSCREEN),
        GetSystemMetrics(SM_CYSCREEN),
        SRCCOPY);

    SelectObject(memdc, oldbmp);
    DeleteDC(memdc);
    DeleteObject(hbitmap);
    ReleaseDC(HWND_DESKTOP, hdc);
}


void screenshot(HDC hdc)
{
    auto w = GetSystemMetrics(SM_CXFULLSCREEN);
    auto h = GetSystemMetrics(SM_CYFULLSCREEN);
    auto hdc = GetDC(HWND_DESKTOP);
    auto hbitmap = CreateCompatibleBitmap(hdc, w, h);
    auto memdc = CreateCompatibleDC(hdc);
    auto oldbmp = SelectObject(memdc, hbitmap);
    BitBlt(memdc, 0, 0, w, h, hdc, 0, 0, SRCCOPY);

    cv::Mat mat(h, w, CV_8UC4);
    BITMAPINFOHEADER bi = { sizeof(bi), w, -h, 1, 32, BI_RGB };
    GetDIBits(hdc, hbitmap, 0, h, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    detector.detect(mat);

    SelectObject(memdc, oldbmp);
    DeleteDC(memdc);
    DeleteObject(hbitmap);
    ReleaseDC(HWND_DESKTOP, hdc);
}*/


int CaptureAnImage(HWND hWnd)
{
    HDC hdcScreen;
    HDC hdcWindow;
    HDC hdcMemDC = NULL;
    HBITMAP hbmScreen = NULL;
    BITMAP bmpScreen;
    DWORD dwBytesWritten = 0;
    DWORD dwSizeofDIB = 0;
    HANDLE hFile = NULL;
    char* lpbitmap = NULL;
    HANDLE hDIB = NULL;
    DWORD dwBmpSize = 0;

    // Retrieve the handle to a display device context for the client 
    // area of the window. 
    hdcScreen = GetDC(NULL);
    hdcWindow = GetDC(hWnd);

    // Create a compatible DC, which is used in a BitBlt from the window DC.
    hdcMemDC = CreateCompatibleDC(hdcWindow);

    if (!hdcMemDC)
    {
        MessageBox(hWnd, L"CreateCompatibleDC has failed", L"Failed", MB_OK);
        goto done;
    }

    // Get the client area for size calculation.
    RECT rcClient;
    GetClientRect(hWnd, &rcClient);

    // This is the best stretch mode.
    SetStretchBltMode(hdcWindow, HALFTONE);

    // The source DC is the entire screen, and the destination DC is the current window (HWND).
    if (!StretchBlt(hdcWindow,
        0, 0,
        rcClient.right, rcClient.bottom,
        hdcScreen,
        0, 0,
        GetSystemMetrics(SM_CXSCREEN),
        GetSystemMetrics(SM_CYSCREEN),
        SRCCOPY))
    {
        MessageBox(hWnd, L"StretchBlt has failed", L"Failed", MB_OK);
        goto done;
    }
    /*
    // Create a compatible bitmap from the Window DC.
    hbmScreen = CreateCompatibleBitmap(hdcWindow, rcClient.right - rcClient.left, rcClient.bottom - rcClient.top);

    if (!hbmScreen)
    {
        MessageBox(hWnd, L"CreateCompatibleBitmap Failed", L"Failed", MB_OK);
        goto done;
    }

    // Select the compatible bitmap into the compatible memory DC.
    SelectObject(hdcMemDC, hbmScreen);

    // Bit block transfer into our compatible memory DC.
    if (!BitBlt(hdcMemDC,
        0, 0,
        rcClient.right - rcClient.left, rcClient.bottom - rcClient.top,
        hdcWindow,
        0, 0,
        SRCCOPY))
    {
        MessageBox(hWnd, L"BitBlt has failed", L"Failed", MB_OK);
        goto done;
    }

    // Get the BITMAP from the HBITMAP.
    GetObject(hbmScreen, sizeof(BITMAP), &bmpScreen);
    */

    // Clean up.
done:
    DeleteObject(hbmScreen);
    DeleteObject(hdcMemDC);
    ReleaseDC(NULL, hdcScreen);
    ReleaseDC(hWnd, hdcWindow);

    return 0;
} 

//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_CREATE:
    {
 
        HANDLE thread = CreateThread(NULL, 0, PaintProc, (LPVOID) hWnd, 0, NULL);
        if (!thread) {
            WaitForSingleObject(thread, INFINITE);
        }
    }
    break;
    case WM_PAINT:
    {
        OutputDebugString(L"paint \n");

        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);
        CaptureAnImage(hWnd);
        //screenshot(hdc);
        EndPaint(hWnd, &ps);
        
    }
    break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
        break;
    }

    return 0;
}

DWORD WINAPI PaintProc(LPVOID lpParam) {
    while (1) {
        OutputDebugString(L"thread \n");
        
        HWND hWnd = (HWND)lpParam;

        SendMessage(hWnd, WM_PAINT, NULL, NULL);
        Sleep(2000);
    }
    return 0;
}