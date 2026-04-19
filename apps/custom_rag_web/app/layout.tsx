import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Custom RAG Streaming Web",
  description: "Next.js client for custom RAG streaming SSE API"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
