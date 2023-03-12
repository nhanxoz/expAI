from docx import Document
from docx.shared import Inches
from docx.enum.style import WD_STYLE_TYPE
import io


def gen_report_docx(trainresult, expName="Bài 5", expID="5234", expCreator="Đỗ Minh Hiếu",
                    expCreatedTime="2022-12-14 17:07:53", dataset="ImageNet", test_dataset_name="LFW", test_dataset_acc=0.9873, ):
    document = Document()

    document.add_heading('Báo cáo bài thí nghiệm', 0)

    p = document.add_paragraph(f'Bài thí nghiệm {expName} | Mã {expID}')

    document.add_paragraph(f'Người tạo: {expCreator}')

    document.add_paragraph(f'Thời gian tạo: {expCreatedTime}')

    document.add_heading('Quá trình huấn luyện', level=1)
    document.add_paragraph('Cấu hình: ', style='Intense Quote')

    document.add_paragraph(
        '{ "weights":"yoloface.pt", "epochs":30, "batch-size":16,"img-size": [640, 640]}	'
    )
    document.add_paragraph('Bộ dữ liệu ', style='Intense Quote')

    document.add_paragraph(
        f'{dataset}'
    )
    document.add_paragraph(
        'Quá trình huấn luyện', style='Intense Quote'
    )

    records = tuple(tuple(d[k] for k in ['trainresultindex', 'lossvalue', 'accuracy']) for d in trainresult)

    table = document.add_table(rows=1, cols=3)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Epoch'
    hdr_cells[0].bold = True
    hdr_cells[1].text = 'Loss'
    hdr_cells[1].bold = True
    hdr_cells[2].text = 'Accuracy'
    hdr_cells[1].bold = True
    for qty, id, desc in records:
        row_cells = table.add_row().cells
        row_cells[0].text = str(qty)
        row_cells[1].text = str(id)
        row_cells[2].text = str(desc)

    document.add_heading(
        f'Đánh giá trên tập dữ liệu: {test_dataset_name}', level=1)
    document.add_paragraph(
        f'Độ chính xác: {test_dataset_acc}'
    )

    file_stream = io.BytesIO()
    document.save(file_stream)
    file_stream.seek(0)
    return file_stream
