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
HWND last;

// Forward declarations of functions included in this code module:
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
DWORD   WINAPI      PaintProc(LPVOID lpParam);

#define MODEL_PATH "cat_seg_mob_traend_5.pt"

class Detector {
private:
    torch::jit::script::Module model;

    std::vector<torch::jit::IValue> preprocessing(cv::Mat src) {
        cv::Mat reszied_img;
        cv::cvtColor(src, src, cv::COLOR_RGBA2RGB);
        cv::resize(src, reszied_img, cv::Size(256, 256), 0, 0, 1);

        auto input = torch::from_blob(reszied_img.data, { reszied_img.rows, reszied_img.cols, reszied_img.channels() }, torch::kUInt8);

        auto mean = torch::tensor({ { 0.485, 0.456, 0.406 } });
        auto std = torch::tensor({ { 0.229, 0.224, 0.225 } });

        input = input / 255;
        input = (input - mean) / std;

        input = input.permute({ 2, 0, 1 }).unsqueeze(0).toType(torch::kFloat32);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        return inputs;
    }

    void draw(cv::Mat src, at::Tensor mask) {

        try {
            cv::Mat mat = cv::Mat(mask.sizes()[0], mask.sizes()[1], CV_8U, mask.data_ptr<uchar>());
            mat = mat * 255;
            cv::resize(mat, mat, cv::Size(src.rows, src.cols), 0, 0, 1);
            cv::cvtColor(mat, mat, cv::COLOR_GRAY2RGBA);

            cv::bitwise_and(src, mat, src);
        }
        catch (cv::Exception e) {
            std::cerr << e.what();
        }
    }

public:
    Detector() {
        model = torch::jit::load(MODEL_PATH);
    }

    void detect(cv::Mat src) {
        std::vector<torch::jit::IValue> inputs = preprocessing(src);
        try {
            
            auto output = model.forward(inputs);
            auto logits = output.toTensor()[0].sigmoid()[0].to(torch::kU8);

            draw(src, logits);
        }
        catch (const c10::Error& e) {
            std::cerr << e.what();
        }
        catch (cv::Exception e) {
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
    last = GetForegroundWindow();

    RECT rcWindow;
    GetWindowRect(last, &rcWindow);

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
        rcWindow.left, rcWindow.top,
        700, 700,
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

void screenshot(HWND hWnd)
{
    RECT rcWindow;
    GetWindowRect(hWnd, &rcWindow);
    auto w = rcWindow.right - rcWindow.left;
    auto h = rcWindow.bottom - rcWindow.top;

    auto hdc = GetDC(last);
    auto hdc2 = GetDC(hWnd);
    auto hbitmap = CreateCompatibleBitmap(hdc, w, h);
    auto memdc = CreateCompatibleDC(hdc);
    auto oldbmp = SelectObject(memdc, hbitmap);
    BitBlt(memdc, 0, 0, w, h, hdc, rcWindow.left+10, rcWindow.top + 32, SRCCOPY);

    cv::Mat mat(h, w, CV_8UC4);
    BITMAPINFOHEADER bi = { sizeof(bi), w, -h, 1, 32, BI_RGB };
    GetDIBits(hdc, hbitmap, 0, h, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    detector.detect(mat);

    SetDIBitsToDevice(hdc2, 0, 0, mat.cols, mat.rows, 0, 0, 0, mat.rows,
        mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

Done:
    SelectObject(memdc, oldbmp);
    DeleteDC(memdc);
    DeleteObject(hbitmap);
    ReleaseDC(HWND_DESKTOP, hdc);
    ReleaseDC(hWnd, hdc2);
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
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);
        screenshot(hWnd);
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
        HWND hWnd = (HWND)lpParam;
        SendMessage(hWnd, WM_PAINT, NULL, NULL);
        Sleep(25);
    }
    return 0;
}