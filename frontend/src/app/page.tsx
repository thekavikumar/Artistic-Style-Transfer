"use client";

import Navbar from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Loader2 } from "lucide-react";
import Image from "next/image";
import { useState } from "react";

export default function Home() {
  const [contentImageUrl, setContentImageUrl] = useState("");
  const [styleImageUrl, setStyleImageUrl] = useState("");
  const [styledImageUrl, setStyledImageUrl] = useState("");
  const [loading, setLoading] = useState(false);

  const handleClick = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://52.172.43.243/style-transfer/?content_url=${encodeURIComponent(
          contentImageUrl
        )}&style_url=${encodeURIComponent(styleImageUrl)}`
      );

      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setStyledImageUrl(imageUrl);
      } else {
        console.error("Error:", response.statusText);
      }
    } catch (error) {
      console.error("Error:", error);
    }
    setLoading(false);
  };

  return (
    <main className="h-screen">
      <Navbar />
      <div className="flex max-w-4xl w-full mx-auto h-full">
        <div className="flex flex-col w-full gap-4 border-r pr-4 pl-2 pt-9">
          <h1 className="text-xl font-semibold ">Welcome to ArtiStyle!</h1>
          <p className="text-lg">
            Add awesome styles to your content images using ArtiStyle. Just
            paste the content image and style image URLs and click on the Style
            button.
          </p>
          <div className="flex items-center gap-3 flex-col">
            <Input
              placeholder="Content Image Url"
              onChange={(e) => setContentImageUrl(e.target.value)}
              value={contentImageUrl}
            />
            <Input
              placeholder="Style Image Url"
              onChange={(e) => setStyleImageUrl(e.target.value)}
              value={styleImageUrl}
            />
            <Button
              className="w-full mt-2"
              onClick={handleClick}
              disabled={loading}
            >
              {loading && <Loader2 className="w-6 h-6 mr-2 animate-spin" />}
              {loading ? "Styling..." : "Style My Image"}
            </Button>
          </div>
        </div>
        <div className="w-full flex flex-col items-center mt-24">
          {styledImageUrl && (
            <div className="mt-4">
              <img src={styledImageUrl} alt="Styled Image" className="mt-2" />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
