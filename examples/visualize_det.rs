//! 可视化检测结果

use image::{GenericImageView, Rgb};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use std::env;

use ocr_rs::{DetOptions, OcrEngine, OcrEngineConfig};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!("用法: visualize_det <det_model> <rec_model> <keys> <image> [output]");
        return;
    }

    let det_model = &args[1];
    let rec_model = &args[2];
    let keys_path = &args[3];
    let image_path = &args[4];
    let output_path = args.get(5).map(|s| s.as_str()).unwrap_or("det_result.png");

    println!("加载模型...");
    let config = OcrEngineConfig::new().with_det_options(DetOptions::fast());

    let engine =
        OcrEngine::new(det_model, rec_model, keys_path, Some(config)).expect("创建引擎失败");

    println!("加载图像: {}", image_path);
    let image = image::open(image_path).expect("加载图像失败");
    let (width, height) = image.dimensions();
    println!("图像尺寸: {}x{}", width, height);

    println!("执行检测...");
    let boxes = engine.detect(&image).expect("检测失败");
    println!("检测到 {} 个边界框", boxes.len());

    // 创建可绘制的图像
    let mut output_image = image.to_rgb8();

    // 颜色列表
    let colors = [
        Rgb([255u8, 0, 0]), // 红
        Rgb([0, 255, 0]),   // 绿
        Rgb([0, 0, 255]),   // 蓝
        Rgb([255, 255, 0]), // 黄
        Rgb([255, 0, 255]), // 品红
        Rgb([0, 255, 255]), // 青
        Rgb([255, 128, 0]), // 橙
        Rgb([128, 0, 255]), // 紫
    ];

    // 绘制每个边界框
    for (i, text_box) in boxes.iter().enumerate() {
        let color = colors[i % colors.len()];
        let rect = Rect::at(text_box.rect.left(), text_box.rect.top())
            .of_size(text_box.rect.width(), text_box.rect.height());

        // 绘制矩形边框（绘制多次使线条更粗）
        draw_hollow_rect_mut(&mut output_image, rect, color);

        // 稍微偏移再画一次，让边框更明显
        if text_box.rect.left() > 0 && text_box.rect.top() > 0 {
            let rect2 = Rect::at(text_box.rect.left() - 1, text_box.rect.top() - 1)
                .of_size(text_box.rect.width() + 2, text_box.rect.height() + 2);
            draw_hollow_rect_mut(&mut output_image, rect2, color);
        }

        // 打印框信息
        println!(
            "[{:2}] ({:4}, {:4}) {:4}x{:3}",
            i,
            text_box.rect.left(),
            text_box.rect.top(),
            text_box.rect.width(),
            text_box.rect.height()
        );
    }

    // 保存结果
    output_image.save(output_path).expect("保存图像失败");
    println!("\n结果已保存到: {}", output_path);
}
