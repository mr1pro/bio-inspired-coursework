// Build the Word version of the report from report/content.json.
//
//   python report/make_content.py && node report/build_docx.js
//
// Layout: US Letter with the margins of the team's template, Calibri 12 pt
// throughout (the coursework asks for Arial/Calibri at 12 pt or larger).

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, ImageRun, Packer, PageBreak, PageNumber,
  Paragraph, ShadingType, Table, TableCell, TableLayoutType, TableRow, TextRun, VerticalAlign,
  WidthType,
} = require("docx");

const HERE = __dirname;
const content = JSON.parse(fs.readFileSync(path.join(HERE, "content.json"), "utf8"));

const FONT = "Calibri";
const SIZE = 24;                    // half-points: 12 pt
const INCH = 1440;                  // twips
const PAGE = { width: 12240, height: 15840 };
const MARGIN = { top: 1080, bottom: 1440, left: 900, right: 900 };   // 0.75, 1, 0.625, 0.625 in
const TEXT_WIDTH = PAGE.width - MARGIN.left - MARGIN.right;          // 7.25 in
const RULE = "595959";

// ---------------------------------------------------------------- inline markup
// **bold**  *italic*  `code`  ^{superscript}  _{subscript}
function runs(text, base = {}) {
  const out = [];
  // A literal asterisk is written {ast} in the content so that it is not
  // mistaken for an italic marker.
  const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\^\{[^}]*\}|_\{[^}]*\})/g;
  for (const piece of text.split(pattern)) {
    if (!piece) continue;
    const options = { font: FONT, size: SIZE, ...base };
    let value = piece;
    if (piece.startsWith("**")) {
      value = piece.slice(2, -2);
      options.bold = true;
    } else if (piece.startsWith("*") && piece.endsWith("*") && piece.length > 2) {
      value = piece.slice(1, -1);
      options.italics = true;
    } else if (piece.startsWith("`")) {
      value = piece.slice(1, -1);
      options.shading = { type: ShadingType.CLEAR, color: "auto", fill: "EFEFEF" };
    } else if (piece.startsWith("^{")) {
      value = piece.slice(2, -1);
      options.superScript = true;
    } else if (piece.startsWith("_{")) {
      value = piece.slice(2, -1);
      options.subScript = true;
    }
    value = value.replace(/\{ast\}/g, "*");
    // keep "p < 0.001", "k = 3", "99.5 ± 0.6" and "5 × 100" on one line
    value = value.replace(/ ([=<±×≈]) /g, "\u00A0$1\u00A0");
    out.push(new TextRun({ text: value, ...options }));
  }
  return out;
}

function bodyParagraph(text, extra = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: 90, line: 240 },
    children: runs(text),
    ...extra,
  });
}

function heading(text) {
  return new Paragraph({
    keepNext: true,
    spacing: { before: 150, after: 70 },
    children: [new TextRun({ text, font: FONT, size: 28, bold: true })],
  });
}

function caption(text, { above }) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    keepNext: above,
    keepLines: true,
    spacing: above ? { before: 80, after: 50 } : { before: 40, after: 130 },
    children: runs(text),
  });
}

// ---------------------------------------------------------------- tables
const NO_BORDER = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const thin = { style: BorderStyle.SINGLE, size: 4, color: RULE };
const thick = { style: BorderStyle.SINGLE, size: 10, color: RULE };

function dataTable(block) {
  const widths = block.columns.map((c) => Math.round(c.width_in * INCH));
  const total = widths.reduce((a, b) => a + b, 0);
  const size = Math.round((block.font_size || 12) * 2);
  const align = (c) => (c.align === "right" ? AlignmentType.RIGHT : AlignmentType.LEFT);

  // keepNext on every row but the last stops a table breaking across pages.
  const cell = (text, column, index, { header, last }) =>
    new TableCell({
      width: { size: widths[index], type: WidthType.DXA },
      verticalAlign: VerticalAlign.TOP,
      margins: { top: 22, bottom: 22, left: 70, right: 70 },
      borders: {
        top: header ? thick : NO_BORDER,
        bottom: header ? thin : last ? thick : NO_BORDER,
        left: NO_BORDER,
        right: NO_BORDER,
      },
      children: [new Paragraph({
        alignment: align(column),
        keepNext: !last,
        spacing: { before: 0, after: 0, line: 240 },
        children: runs(text, { size, bold: header ? true : undefined }),
      })],
    });

  const rows = [
    new TableRow({
      tableHeader: true,
      cantSplit: true,
      children: block.columns.map((c, i) => cell(c.header, c, i, { header: true, last: false })),
    }),
    ...block.rows.map((row, r) => new TableRow({
      cantSplit: true,
      children: row.map((text, i) => cell(text, block.columns[i], i,
        { header: false, last: r === block.rows.length - 1 })),
    })),
  ];
  return [
    caption(block.caption, { above: true }),
    new Table({
      width: { size: total, type: WidthType.DXA },
      columnWidths: widths,
      layout: TableLayoutType.FIXED,
      alignment: AlignmentType.CENTER,
      rows,
    }),
    new Paragraph({ spacing: { before: 0, after: 110, line: 120 }, children: [] }),
  ];
}

// ---------------------------------------------------------------- figures
function pngSize(file) {
  const buffer = fs.readFileSync(file);
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20), buffer };
}

function figure(block) {
  const { width, height, buffer } = pngSize(block.file);
  const widthPx = Math.round(block.width_in * 96);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: 60, after: 0 },
      children: [new ImageRun({
        type: "png",
        data: buffer,
        transformation: { width: widthPx, height: Math.round((widthPx * height) / width) },
        altText: { title: "Figure", description: block.caption.replace(/[*`]/g, ""), name: path.basename(block.file) },
      })],
    }),
    caption(block.caption, { above: false }),
  ];
}

function render(blocks) {
  const out = [];
  for (const block of blocks) {
    if (block.type === "h1") out.push(heading(block.text));
    else if (block.type === "p") out.push(bodyParagraph(block.text));
    else if (block.type === "table") out.push(...dataTable(block));
    else if (block.type === "figure") out.push(...figure(block));
  }
  return out;
}

// ---------------------------------------------------------------- title block
function titleBlock(front) {
  const logo = pngSize(front.logo);
  const logoWidth = 100;
  const line = (text, options = {}, spacing = { before: 0, after: 20 }) =>
    new Paragraph({ spacing, children: [new TextRun({ text, font: FONT, size: SIZE, ...options })] });
  const leftWidth = Math.round(1.2 * INCH);
  const cellBorders = { top: NO_BORDER, bottom: NO_BORDER, left: NO_BORDER, right: NO_BORDER };
  return [
    new Table({
      width: { size: TEXT_WIDTH, type: WidthType.DXA },
      columnWidths: [leftWidth, TEXT_WIDTH - leftWidth],
      layout: TableLayoutType.FIXED,
      rows: [new TableRow({
        children: [
          new TableCell({
            width: { size: leftWidth, type: WidthType.DXA },
            verticalAlign: VerticalAlign.CENTER,
            borders: cellBorders,
            margins: { top: 0, bottom: 0, left: 0, right: 120 },
            children: [new Paragraph({
              children: [new ImageRun({
                type: "png",
                data: logo.buffer,
                transformation: { width: logoWidth, height: Math.round((logoWidth * logo.height) / logo.width) },
                altText: { title: "Logo", description: "Heriot-Watt University logo", name: "logo" },
              })],
            })],
          }),
          new TableCell({
            width: { size: TEXT_WIDTH - leftWidth, type: WidthType.DXA },
            verticalAlign: VerticalAlign.CENTER,
            borders: cellBorders,
            margins: { top: 0, bottom: 0, left: 60, right: 0 },
            children: [
              line(front.campus),
              line(front.course),
              line(front.title, { bold: true, size: 32 }, { before: 40, after: 40 }),
              line(front.authors, { bold: true }),
              line(front.code, {}, { before: 0, after: 0 }),
            ],
          }),
        ],
      })],
    }),
    new Paragraph({
      spacing: { before: 60, after: 60, line: 120 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: RULE, space: 1 } },
      children: [],
    }),
  ];
}

// ---------------------------------------------------------------- document
const referenceParagraphs = content.references.map((text) => new Paragraph({
  alignment: AlignmentType.LEFT,
  indent: { left: 432, hanging: 432 },
  spacing: { before: 0, after: 90, line: 240 },
  children: runs(text),
}));

const children = [
  ...titleBlock(content.front),
  ...render(content.body),
  new Paragraph({ children: [new PageBreak()] }),
  heading("References"),
  ...referenceParagraphs,
  new Paragraph({ children: [new PageBreak()] }),
  ...render(content.appendix),
];

const document = new Document({
  creator: "Ahmed Abdallah and Mohamed Ali",
  title: content.front.title,
  description: "F21BC Biologically Inspired Computation coursework report",
  styles: { default: { document: { run: { font: FONT, size: SIZE } } } },
  sections: [{
    properties: { page: { size: PAGE, margin: MARGIN } },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SIZE })],
        })],
      }),
    },
    children,
  }],
});

const target = path.join(HERE, process.argv[2] || "F21BC_Coursework_Report.docx");
Packer.toBuffer(document).then((buffer) => {
  fs.writeFileSync(target, buffer);
  console.log("wrote", target);
});
