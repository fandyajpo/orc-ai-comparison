import { ImageAnnotatorClient } from "@google-cloud/vision";
import OpenAI from "openai";
import axios from "axios";
import sharp from "sharp";

export type ParseUsage = {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  reasoning_tokens: number;
};

export type Timing = {
  fetchMs: number;
  compressMs: number;
  ocrMs: number;
  parseMs: number;
  totalMs: number;
};

// Input bisa url, base64, atau file
export type ImageInput =
  | { type: "url"; value: string }
  | { type: "base64"; value: string; mimeType?: string }
  | { type: "file"; value: Buffer | Uint8Array };

const openai = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
});

const PASSPORT_SCHEMA = {
  type: "object",
  properties: {
    birthDate: {
      type: "object",
      properties: { value: { type: ["string", "null"] } },
      required: ["value"],
      additionalProperties: false,
    },
    birthPlace: {
      type: "object",
      properties: { value: { type: ["string", "null"] } },
      required: ["value"],
      additionalProperties: false,
    },
    country: {
      type: "object",
      properties: { value: { type: ["string", "null"] } },
      required: ["value"],
      additionalProperties: false,
    },
    expiryDate: {
      type: "object",
      properties: { value: { type: ["string", "null"] } },
      required: ["value"],
      additionalProperties: false,
    },
    gender: {
      type: "object",
      properties: { value: { type: ["string", "null"] } },
      required: ["value"],
      additionalProperties: false,
    },
    givenNames: {
      type: "array",
      items: {
        type: "object",
        properties: { value: { type: ["string", "null"] } },
        required: ["value"],
        additionalProperties: false,
      },
    },
    idNumber: {
      type: "object",
      properties: { value: { type: ["string", "null"] } },
      required: ["value"],
      additionalProperties: false,
    },
    issuanceDate: {
      type: "object",
      properties: { value: { type: ["string", "null"] } },
      required: ["value"],
      additionalProperties: false,
    },
    mrz1: {
      type: "object",
      properties: { value: { type: ["string", "null"] } },
      required: ["value"],
      additionalProperties: false,
    },
    mrz2: {
      type: "object",
      properties: { value: { type: ["string", "null"] } },
      required: ["value"],
      additionalProperties: false,
    },
    surname: {
      type: "object",
      properties: { value: { type: ["string", "null"] } },
      required: ["value"],
      additionalProperties: false,
    },
  },
  required: [
    "birthDate",
    "birthPlace",
    "country",
    "expiryDate",
    "gender",
    "givenNames",
    "idNumber",
    "issuanceDate",
    "mrz1",
    "mrz2",
    "surname",
  ],
  additionalProperties: false,
} as const;
export type PassportData = {
  birthDate: { value: string | null };
  birthPlace: { value: string | null };
  country: { value: string | null };
  expiryDate: { value: string | null };
  gender: { value: string | null };
  givenNames: { value: string | null }[];
  idNumber: { value: string | null };
  issuanceDate: { value: string | null };
  mrz1: { value: string | null };
  mrz2: { value: string | null };
  surname: { value: string | null };
};

export class GoogleVisionService {
  private client: ImageAnnotatorClient;

  constructor() {
    if (!process.env.GOOGLE_CREDENTIALS_BASE64) {
      throw new Error("GOOGLE_CREDENTIALS_BASE64 env var is not set");
    }
    const credentials = JSON.parse(
      Buffer.from(process.env.GOOGLE_CREDENTIALS_BASE64!, "base64").toString(
        "utf-8",
      ),
    );
    this.client = new ImageAnnotatorClient({ credentials });
  }

  // ✅ Method utama — terima semua jenis input
  async extract(input: ImageInput): Promise<{
    raw: string;
    passport: PassportData;
    usage: ParseUsage;
    timing: Timing;
  }> {
    const total = Date.now();

    // 1. Resolve ke Buffer
    const t1 = Date.now();
    const rawBuffer = await this.resolveToBuffer(input);
    const fetchMs = Date.now() - t1;

    // 2. Compress
    const t2 = Date.now();
    const compressed = await sharp(rawBuffer)
      .resize(1600, undefined, { withoutEnlargement: true })
      .sharpen()
      .jpeg({ quality: 85 })
      .toBuffer();
    const compressMs = Date.now() - t2;

    // 3. OCR via Google Vision
    const t3 = Date.now();
    const [result] = await this.client.documentTextDetection({
      image: { content: compressed.toString("base64") },
    });
    const rawText = result.fullTextAnnotation?.text ?? "";
    const ocrMs = Date.now() - t3;

    // 4. Parse via GPT
    const t4 = Date.now();
    const gpt = await this.parsePassportData(rawText);
    const parseMs = Date.now() - t4;

    return {
      raw: rawText,
      passport: gpt.passport,
      usage: gpt.usage,
      timing: {
        fetchMs,
        compressMs,
        ocrMs,
        parseMs,
        totalMs: Date.now() - total,
      },
    };
  }

  // ✅ Shorthand methods biar lebih ergonomis
  async extractFromUrl(url: string) {
    return this.extract({ type: "url", value: url });
  }

  async extractFromBase64(base64: string, mimeType?: string) {
    return this.extract({ type: "base64", value: base64, mimeType });
  }

  async extractFromFile(buffer: Buffer | Uint8Array) {
    return this.extract({ type: "file", value: buffer });
  }

  // ✅ Resolve semua jenis input ke Buffer
  private async resolveToBuffer(input: ImageInput): Promise<Buffer> {
    switch (input.type) {
      case "url": {
        const response = await axios.get<ArrayBuffer>(input.value, {
          responseType: "arraybuffer",
        });
        return Buffer.from(response.data);
      }

      case "base64": {
        // Handle kalau ada prefix "data:image/jpeg;base64,..."
        const raw = input.value.includes(",")
          ? input.value.split(",")[1]
          : input.value;
        return Buffer.from(raw, "base64");
      }

      case "file": {
        return Buffer.isBuffer(input.value)
          ? input.value
          : Buffer.from(input.value);
      }
    }
  }

  private async parsePassportData(
    rawText: string,
  ): Promise<{ passport: PassportData; usage: ParseUsage }> {
    const response = await openai.chat.completions.create({
      model: "openai/gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: `Extract passport information.

Rules:
- Dates MUST use YYYY-MM-DD
- country MUST use ISO alpha-3
- givenNames is array
- mrz1 and mrz2 split lines
- Return null when unclear`,
        },
        {
          role: "user",
          content: rawText,
        },
      ],
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "passport",
          strict: true,
          schema: PASSPORT_SCHEMA,
        },
      },
    });

    const content = response.choices[0].message.content;
    if (!content) throw new Error("Empty response from GPT");

    return {
      passport: JSON.parse(content) as PassportData,
      usage: {
        input_tokens: response.usage?.prompt_tokens ?? 0,
        output_tokens: response.usage?.completion_tokens ?? 0,
        total_tokens: response.usage?.total_tokens ?? 0,
        reasoning_tokens:
          response.usage?.completion_tokens_details?.reasoning_tokens ?? 0,
      },
    };
  }
}
