import OpenAI from "openai";
import axios from "axios";
import sharp from "sharp";
import { PassportSchema } from "../schemas/passport.schema";

export type Timing = {
  fetchMs: number;
  compressMs: number;
  extractMs: number;
  parseMs: number;
  totalMs: number;
};

export class PassportAiService {
  private openai: OpenAI;

  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }

  async extract(imageUrl: string) {
    const totalStart = Date.now();

    // 1. Fetch image
    const fetchStart = Date.now();
    const response = await axios.get<ArrayBuffer>(imageUrl, {
      responseType: "arraybuffer",
      timeout: 30000,
    });
    const fetchMs = Date.now() - fetchStart;

    // 2. Compress image
    const compressStart = Date.now();
    const compressed = await sharp(Buffer.from(response.data))
      .resize(1600, undefined, { withoutEnlargement: true }) // max 1600px width, skip if smaller
      .jpeg({ quality: 85 })
      .toBuffer();
    const imageBase64 = compressed.toString("base64");
    const compressMs = Date.now() - compressStart;

    // 3. GPT extract
    const extractStart = Date.now();
    const result = await this.openai.responses.create({
      model: "gpt-4.1-mini",
      text: {
        format: {
          type: "json_schema",
          name: "passport_extract",
          schema: {
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
          },
        },
      },
      input: [
        {
          role: "system",
          content: `
Extract passport information.

Rules:
- Dates MUST use YYYY-MM-DD
- country MUST use ISO alpha-3
- givenNames is array
- mrz1 and mrz2 split lines
- Return null when unclear
`,
        },
        {
          role: "user",
          content: [
            { type: "input_text", text: "Extract passport information." },
            {
              type: "input_image",
              image_url: `data:image/jpeg;base64,${imageBase64}`,
              detail: "high",
            },
          ],
        },
      ],
    });
    const extractMs = Date.now() - extractStart;

    // 4. Parse & validate
    const parseStart = Date.now();
    const parsed = JSON.parse(result.output_text);
    const data = PassportSchema.parse(parsed);
    const parseMs = Date.now() - parseStart;

    const usage = result.usage;

    return {
      data,
      usage: {
        input_tokens: usage?.input_tokens ?? 0,
        output_tokens: usage?.output_tokens ?? 0,
        total_tokens: usage?.total_tokens ?? 0,
        reasoning_tokens: usage?.output_tokens_details?.reasoning_tokens ?? 0,
      },
      timing: {
        fetchMs,
        compressMs,
        extractMs,
        parseMs,
        totalMs: Date.now() - totalStart,
      } satisfies Timing,
    };
  }
}
