/**
 * Generates a concise title from the first user message
 * Extracts key words or truncates to a reasonable length
 */
export function generateTitleFromMessage(message: string): string {
  // Remove extra whitespace and normalize
  const cleaned = message.trim().replace(/\s+/g, " ");

  // If message is short enough, use it as is
  if (cleaned.length <= 50) {
    return cleaned;
  }

  // Try to find a good breaking point (sentence end or comma)
  const sentenceMatch = cleaned.match(/^([^.!?\n]+)[.!?]/);
  if (sentenceMatch && sentenceMatch[1].length <= 50) {
    return sentenceMatch[1].trim();
  }

  // Try to break at a comma
  const commaIndex = cleaned.indexOf(",");
  if (commaIndex > 0 && commaIndex <= 50) {
    return cleaned.substring(0, commaIndex).trim();
  }

  // Break at word boundary near 50 characters
  const truncated = cleaned.substring(0, 50);
  const lastSpace = truncated.lastIndexOf(" ");
  if (lastSpace > 20) {
    return truncated.substring(0, lastSpace).trim() + "...";
  }

  return truncated.trim() + "...";
}

