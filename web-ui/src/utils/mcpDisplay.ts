/** MCP Server / 工具中文备注（与后端 mcp_zh_note 对齐，供 UI 展示）。 */

const MCP_SERVER_ZH: Record<string, string> = {
  'mysql-biz': '业务库测试环境（RDS / dfzx_cbs_test，只读）',
  'mysql-bigdata': '大数据平台测试库（PolarDB MySQL，只读）',
  mysql: 'MySQL 数据库（只读）',
  postgres: 'PostgreSQL 数据库',
  filesystem: '工作区文件系统',
};

const MCP_TOOL_ZH: Record<string, string> = {
  mysql_query: '执行 SQL 查询；多库模式请用「库名.表名」，勿写库',
  query: '执行 SQL 查询',
  read_query: '执行只读 SQL 查询',
  read_file: '读取文件内容',
  read_text_file: '读取文本文件',
  list_directory: '列出目录下的文件与子目录',
  directory_tree: '查看目录树结构',
  search_files: '按名称搜索文件',
  get_file_info: '查看文件元信息',
  list_allowed_directories: '列出允许访问的根目录',
};

/** 从 mcp__server__tool 解析 server / tool 名。 */
export function parseMcpToolName(fullName: string): { server: string; tool: string } | null {
  if (!fullName.startsWith('mcp__')) {
    return null;
  }
  const rest = fullName.slice('mcp__'.length);
  const idx = rest.indexOf('__');
  if (idx <= 0) {
    return null;
  }
  return { server: rest.slice(0, idx), tool: rest.slice(idx + 2) };
}

export function mcpServerZh(server: string): string {
  return MCP_SERVER_ZH[server] || `外部 MCP「${server}」`;
}

export function mcpToolZhNote(fullName: string): string {
  const parsed = parseMcpToolName(fullName);
  if (!parsed) {
    return '外部 MCP 工具';
  }
  const srv = mcpServerZh(parsed.server);
  const tool = MCP_TOOL_ZH[parsed.tool] || `调用工具「${parsed.tool}」`;
  return `${srv} · ${tool}`;
}

/** 展示用：中文备注 + 原文描述（若描述已含中文说明则不重复）。 */
export function formatMcpToolDisplay(name: string, description?: string): string {
  const note = mcpToolZhNote(name);
  const raw = (description || '').trim();
  if (!raw) {
    return note;
  }
  if (raw.includes('【中文说明】') || raw.startsWith(note)) {
    return raw;
  }
  return `${note}\n${raw}`;
}
