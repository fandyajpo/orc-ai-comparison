import { z } from "zod";

const StringFieldSchema = z.object({
  value: z.string().nullable(),
});

const DateFieldSchema = z.object({
  value: z.string().nullable(),
});

export const PassportSchema = z.object({
  birthDate: DateFieldSchema,
  birthPlace: StringFieldSchema,
  country: StringFieldSchema,
  expiryDate: DateFieldSchema,
  gender: StringFieldSchema,
  givenNames: z.array(StringFieldSchema),
  idNumber: StringFieldSchema,
  issuanceDate: DateFieldSchema,
  mrz1: StringFieldSchema,
  mrz2: StringFieldSchema,
  surname: StringFieldSchema,
});

export type PassportExtraction = z.infer<typeof PassportSchema>;
