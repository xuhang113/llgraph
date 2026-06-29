export interface ChatImageAttachment {
  media_type: string;
  /** 附件预览 URL（image_ref） */
  url?: string;
  id?: string;
}

/** 输入框待发送图片（multipart 直传 File） */
export interface ComposerImage {
  id: string;
  name: string;
  media_type: string;
  preview_url: string;
  file: File;
}

export const MAX_COMPOSER_IMAGES = 50;
export const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
export const ACCEPT_IMAGE_TYPES = 'image/png,image/jpeg,image/webp,image/gif';
export const IMAGE_ATTACH_TOOLTIP = '上传图片（支持 PNG、JPEG、WebP、GIF）';
