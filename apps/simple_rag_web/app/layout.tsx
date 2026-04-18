import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Simple RAG Web",
  description: "Next.js client for the OCI simple RAG HTTP API"
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
