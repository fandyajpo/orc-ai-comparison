import "dotenv/config";
import { GoogleVisionService } from "./services/google-vision.service";
import { PassportAiService } from "./services/passport-ai.service";

async function main() {
  const service = new GoogleVisionService();

  const result = await service.extractFromUrl(
    "https://media.squaremetre.io/DocumentLicence/fandyglitch9@gmail.comPassportUploadByLink_Ovz0pIfu",
  );

  console.dir(result, {
    depth: null,
  });
}

main().catch(console.error);
